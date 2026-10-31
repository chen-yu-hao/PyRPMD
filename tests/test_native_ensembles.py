"""Native ensemble contracts, independent equations and statistical controls.

Analytic models test the numerical implementation, not chemical-rate accuracy
or transferable PIGLET parameters. No real-system validation is implied.
"""
from __future__ import annotations

import copy
import cProfile
import ctypes
from pathlib import Path
import shutil
import subprocess
import threading
import time

import numpy as np
import pytest
from scipy.linalg import expm

pytest.importorskip("pyrpmd._thermal")
from pyrpmd.native_ensemble import NativeEnsemble, NativePeriodicPES, _state_digest


def make(*, beads=4, thermostat="pile-l", barostat="none", seed=41, **kwargs):
    args = dict(q=np.random.default_rng(8).normal(scale=.1, size=(beads, 2, 3)),
                masses=[1., 2.], beta=2., dt=.025, seed=seed,
                potential={"kind": "harmonic", "k": .7},
                thermostat=thermostat, barostat=barostat)
    args.update(kwargs)
    return NativeEnsemble(**args)


def gle(mode, beads):
    spec = {"A": [[1., 2.], [-2., 1.]], "C": np.eye(2).tolist()}
    if mode == "piglet":
        spec.update(A=np.repeat(np.array(spec["A"])[None], beads, axis=0).tolist(),
                    beta=2., beads=beads, provenance="TEST: canonical identity, NOT a calibrated PIGLET set")
    return {"mode": mode, "gle": spec}


THERMOSTATS = ["none", "andersen", "langevin", "pile-l", "pile-g", "svr",
              "gle", "nm-gle", "nm-gle-g", "piglet", "trpmd", "ffl"]


@pytest.mark.parametrize("mode", THERMOSTATS)
@pytest.mark.parametrize("beads", [1, 3, 4])
def test_exact_block_and_restart_identity(mode, beads, tmp_path):
    thermostat = gle(mode, beads) if "gle" in mode or mode == "piglet" else mode
    whole, split = make(beads=beads, thermostat=thermostat), make(beads=beads, thermostat=thermostat)
    expected = whole.run(103, stride=7, record_centroids=True)
    blocks = [split.run(8, stride=7, record_centroids=True),
              split.run(35, stride=7, record_centroids=True)]
    checkpoint = tmp_path / "checkpoint.npz"
    split.save(checkpoint)
    resumed = make(beads=beads, thermostat=thermostat, seed=999)
    resumed.load(checkpoint)
    blocks.append(resumed.run(60, stride=7, record_centroids=True))
    np.testing.assert_array_equal(np.concatenate([r["samples"] for r in blocks]), expected["samples"])
    np.testing.assert_array_equal(np.concatenate([r["centroids"] for r in blocks]), expected["centroids"])
    assert resumed.state()["digest"] == whole.state()["digest"]
    assert expected["force_evaluations"] == 104
    assert expected["python_step_callbacks"] == 0


@pytest.mark.parametrize("mode", ["bzp", "mttk", "rgb", "scr"])
def test_npt_exact_restart_and_streaming(mode, tmp_path):
    b = {"mode": mode, "tau": 1.5}
    if mode == "rgb":
        b.update(stress=np.eye(3).tolist(), reference_cell=(np.eye(3)*2).tolist())
    else:
        b["pressure"] = .7
    if mode == "scr":
        b["compressibility"] = .3
    if mode in ("rgb", "mttk"):
        b["cell_mask"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    settings = dict(beads=1 if mode == "scr" else 3, barostat=b,
                    cell=np.eye(3)*2, potential={"kind": "free"}, thermostat="pile-l")
    a, z = make(**settings), make(**settings)
    expected = a.run(150, stride=7, record_centroids=True)
    first = z.run(41, stride=7, record_centroids=True, trajectory=tmp_path/"first.npy")
    state = z.state()
    resumed = make(**settings)
    resumed.restore(state)
    second = resumed.run(109, stride=7, record_centroids=True, trajectory=tmp_path/"second.npy")
    assert isinstance(first["samples"], np.memmap)
    np.testing.assert_array_equal(np.concatenate([first["samples"], second["samples"]]), expected["samples"])
    np.testing.assert_array_equal(np.concatenate([first["centroids"], second["centroids"]]), expected["centroids"])
    assert resumed.state()["digest"] == a.state()["digest"]


def test_stream_zero_frames_and_no_overwrite(tmp_path):
    e = make()
    r = e.run(0, trajectory=tmp_path/"empty.npy")
    assert r["samples"].shape == (0, 18)
    state = e.state()
    with pytest.raises(FileExistsError):
        e.run(100, trajectory=tmp_path/"empty.npy")
    assert e.state()["digest"] == state["digest"]
    assert not list(tmp_path.glob("*.tmp"))


def test_no_silent_python_fallback():
    class PythonOnly:
        def evaluate(self, q):
            raise AssertionError("Python PES must never be called")
    with pytest.raises((ValueError, ImportError, ModuleNotFoundError)):
        make(potential=PythonOnly())


@pytest.mark.parametrize("mode", ["pile-l", "pile-g", "svr", "andersen", "langevin", "ffl", "nm-gle", "piglet"])
def test_fixed_atoms_are_exactly_pinned(mode):
    thermostat = gle(mode, 4) if "gle" in mode or mode == "piglet" else mode
    q = np.ones((4, 2, 3))*.2
    e = make(q=q, thermostat=thermostat, fixed_atoms=[1])
    e.run(500, stride=100)
    state = e.state()
    np.testing.assert_array_equal(state["q"][:, 1], q[:, 1])
    np.testing.assert_array_equal(state["p"][:, 1], 0)


def test_trpmd_centroid_is_unthermostatted():
    q = np.zeros((8, 2, 3))
    p = np.repeat(np.array([[[.2, -.3, .1], [.6, -.1, .4]]]), 8, axis=0)
    e = make(beads=8, q=q, p=p, thermostat="trpmd", potential={"kind": "free"})
    r = e.run(2000, stride=100, record_centroids=True)
    state = e.state()
    np.testing.assert_allclose(state["p"].mean(0), p.mean(0), atol=5e-12, rtol=0)
    expected = (2000*.025*p.mean(0))/np.array([1., 2.])[:, None]
    np.testing.assert_allclose(state["q"].mean(0), expected, atol=1e-10, rtol=0)
    assert r["thermostat"] == "trpmd"


@pytest.mark.parametrize("integrator", ["cayley-bcb", "normal-mode-bab"])
@pytest.mark.parametrize("beads", [1, 3, 8, 32])
def test_free_ring_matches_existing_dynamics(integrator, beads):
    dynamics_module = pytest.importorskip("pyrpmd.dynamics")
    builtin = pytest.importorskip("pyrpmd.potentials.builtin")
    rng = np.random.default_rng(37)
    q, p = rng.normal(scale=.1, size=(2, beads, 2, 3))
    e = make(beads=beads, q=q, p=p, thermostat="none", integrator=integrator)
    dyn = dynamics_module.Dynamics(builtin.HarmonicPotential(k=.7), np.array([1., 2.]),
                                   2., beads, .025, integrator=integrator, native=False)
    reference = dynamics_module.State(q.copy(), p.copy())
    for _ in range(50):
        dyn.step(reference)
    e.run(50)
    np.testing.assert_allclose(e.state()["q"], reference.q, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(e.state()["p"], reference.p, rtol=2e-10, atol=2e-11)


def test_svr_zero_kinetic_energy_is_finite():
    e = make(p=np.zeros((4, 2, 3)), potential={"kind": "free"}, thermostat="svr")
    e.run(1)
    p = e.state()["p"]
    assert np.isfinite(p).all() and np.any(p != 0)


def test_gle_joint_stationary_covariance_and_scalar_limit():
    atoms = 6000
    thermo = {"mode": "gle", "gle": {"A": [[1., 2.], [-2., 1.]], "C": [[1., 0.], [0., 1.]]}}
    e = NativeEnsemble(np.zeros((1, atoms, 3)), np.ones(atoms), 2., .1,
                       potential={"kind": "free"}, thermostat=thermo, seed=81)
    e.run(20, stride=20)
    state = e.state()
    values = np.array([state["p"].ravel()*np.sqrt(2), state["aux"].ravel()])
    np.testing.assert_allclose(np.cov(values), np.eye(2), atol=.035, rtol=0)
    a = make(thermostat={"mode": "langevin", "gamma": .7})
    b = make(thermostat={"mode": "gle", "gle": {"A": [[.7]], "C": [[1.]]}})
    a.run(300);b.run(300)
    np.testing.assert_allclose(a.state()["q"], b.state()["q"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(a.state()["p"], b.state()["p"], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("beads", [1, 3, 4, 7, 8])
@pytest.mark.parametrize("mode", ["pile-l", "langevin", "andersen", "nm-gle"])
def test_harmonic_finite_bead_equilibrium(beads, mode):
    t = gle(mode, beads) if "gle" in mode else mode
    e = NativeEnsemble(np.zeros((beads, 1, 3)), [1.], 4., .05,
                       potential={"kind": "harmonic", "k": 1.}, thermostat=t, seed=812)
    r = e.run(100000, stride=10)
    u = r["samples"][2000:, 2]
    omega = 2*beads/4*np.sin(np.pi*np.arange(beads)/beads)
    target = 1.5/4*np.sum(1/(1+omega*omega))
    batch = np.array_split(u, 16)
    error = np.std([x.mean() for x in batch], ddof=1)/4
    assert abs(u.mean()-target) < 6*error + .01*target


@pytest.mark.parametrize("mode,beads", [("bzp", 1), ("bzp", 4), ("mttk", 1), ("mttk", 4), ("scr", 1)])
def test_ideal_gas_npt_volume_distribution(mode, beads):
    b = {"mode": mode, "pressure": 1., "tau": .5}
    if mode == "mttk":
        b["cell_mask"] = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    if mode == "scr":
        b["compressibility"] = 1.
    else:
        b["gamma"] = 2.
    e = NativeEnsemble(np.zeros((beads, 8, 3)), np.ones(8), 1., .005,
                       potential={"kind": "free"}, cell=np.eye(3)*9**(1/3),
                       thermostat={"mode": "langevin", "gamma": 2.}, barostat=b, seed=812)
    v = e.run(300000, stride=20)["samples"][1500:, 6]
    batches = np.array_split(v, 20)
    mean_error = np.std([x.mean() for x in batches], ddof=1)/np.sqrt(20)
    var_error = np.std([np.mean((x-9)**2) for x in batches], ddof=1)/np.sqrt(20)
    # p(V) proportional to V^N exp(-beta*P_ext*V), including the +1.
    assert abs(v.mean()-9.) < 6*mean_error + .03
    assert abs(np.mean((v-9.)**2)-9.) < 6*var_error + .1


@pytest.mark.parametrize("mode", ["bzp", "mttk", "rgb"])
def test_independent_constant_pressure_one_step(mode):
    q = np.array([[[.1, .2, -.3], [.3, -.2, .5]]])
    p = np.array([[[.4, -.1, .2], [-.2, .6, .5]]])
    cell = np.array([[2., 0, 0], [.2, 2.1, 0], [.1, -.1, 2.2]])
    b = {"mode": mode, "tau": .8, "gamma": 0.}
    if mode == "rgb":
        b.update(stress=np.diag([.4, .5, .6]).tolist(), reference_cell=cell.tolist())
    else:
        b["pressure"] = .4
    e = make(beads=1, q=q, p=p, cell=cell, potential={"kind": "free"},
             thermostat="none", barostat=b)
    initial = e.state()
    # Pure-shear initial velocity tests the repeated-eigenvalue/Jordan case.
    initial["cell_p"][0, 0 if mode == "bzp" else 1] = .25
    initial["digest"] = _state_digest(initial)
    e.restore(initial)
    mass = np.array([1., 2.])
    theta, dt = .5, .025
    W = (3 if mode == "bzp" else 1)*2*theta*.8**2
    qp, pp, hp, piston = q[0].copy(), p[0].copy(), cell.copy(), initial["cell_p"].copy()
    invref = np.linalg.inv(cell.T)
    def pkick():
        volume = np.linalg.det(hp)
        kinetic = np.diag(np.sum(pp**2/mass[:, None], axis=0))
        if mode == "bzp":
            piston[0, 0] += dt/2*(np.trace(kinetic)-3*.4*volume+3*theta)
        else:
            if mode == "rgb":
                G = hp.T @ invref
                ext = np.linalg.det(cell)*G @ np.diag([.4, .5, .6]) @ G.T
            else:
                ext = .4*volume*np.eye(3)
            piston[:] += dt/2*np.triu(kinetic-ext+theta*np.diag([3., 2., 1.]))
    pkick()
    velocity = np.eye(3)*piston[0, 0]/W if mode == "bzp" else piston/W
    for atom in range(2):
        L = np.block([[velocity, np.eye(3)/mass[atom]], [np.zeros((3, 3)), -velocity]])
        value = expm(dt*L) @ np.r_[qp[atom], pp[atom]]
        qp[atom], pp[atom] = value[:3], value[3:]
    hp = hp @ expm(dt*velocity).T
    pkick()
    e.run(1)
    state = e.state()
    np.testing.assert_allclose(state["q"][0], qp, rtol=3e-13, atol=3e-14)
    np.testing.assert_allclose(state["p"][0], pp, rtol=3e-13, atol=3e-14)
    np.testing.assert_allclose(state["cell"], hp, rtol=3e-13, atol=3e-14)
    np.testing.assert_allclose(state["cell_p"], piston, rtol=3e-13, atol=3e-14)


@pytest.mark.parametrize("mode", ["bzp", "mttk", "rgb"])
def test_npt_deterministic_time_reversal(mode):
    b = {"mode": mode, "tau": 1., "gamma": 0.}
    if mode == "rgb":
        b.update(stress=np.eye(3).tolist(), reference_cell=(np.eye(3)*2).tolist())
    else:
        b["pressure"] = .5
    e = make(beads=4, barostat=b, cell=np.eye(3)*2, potential={"kind": "free"}, thermostat="none")
    initial = e.state()
    e.run(50)
    reverse = e.state()
    reverse["p"] *= -1
    reverse["cell_p"] *= -1
    reverse["digest"] = _state_digest(reverse)
    e.restore(reverse)
    e.run(50)
    state = e.state()
    np.testing.assert_allclose(state["q"], initial["q"], atol=2e-11, rtol=0)
    np.testing.assert_allclose(state["p"], -initial["p"], atol=2e-11, rtol=0)
    np.testing.assert_allclose(state["cell"], initial["cell"], atol=2e-11, rtol=0)
    np.testing.assert_allclose(state["cell_p"], -initial["cell_p"], atol=2e-11, rtol=0)


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler unavailable")
    path = tmp_path_factory.mktemp("thermal-pes")
    source = path/"pes.c"
    source.write_text(r'''
#include <stdint.h>
#include <stdio.h>
static int calls=0, failure=-1;
void set_failure(int n) {calls=0; failure=n;}
int get_calls(void) {return calls;}
int32_t evaluate(void*ctx,int64_t P,int64_t N,const double*q,const double*h,
                 const int32_t*pbc,double*e,double*f,double*v,char*error,int64_t cap) {
    (void)ctx;(void)h;(void)pbc;
    ++calls;
    if(calls==failure){snprintf(error,(size_t)cap,"injected native failure");return 1;}
    for(int64_t b=0;b<P;++b){
        e[b]=0;
        for(int j=0;j<9;++j)v[9*b+j]=0;
        for(int64_t j=0;j<3*N;++j){int64_t i=b*3*N+j;e[b]+=.5*q[i]*q[i];f[i]=-q[i];}
    }
    return 0;
}
''')
    library = path/"pes.so"
    subprocess.run([cc, "-O2", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    return library


def test_compiled_callback_no_python_steps_and_rollback(native_library, tmp_path):
    potential = NativePeriodicPES(native_library, "evaluate", 2)
    control = ctypes.CDLL(str(native_library))
    control.set_failure(-1)
    e = make(potential=potential)
    before = e.state()
    control.set_failure(12)
    with pytest.raises(RuntimeError, match="injected native failure"):
        e.run(100, stride=3, trajectory=tmp_path/"failed.npy")
    assert e.state()["digest"] == before["digest"]
    assert not (tmp_path/"failed.npy").exists()
    assert not list(tmp_path.iterdir())
    control.set_failure(-1)
    profile = cProfile.Profile()
    profile.enable()
    result = e.run(10000, stride=1000)
    profile.disable()
    assert result["force_evaluations"] == 10001
    assert control.get_calls() == 10000
    stats = profile.getstats()
    assert sum(item.callcount for item in stats if isinstance(item.code, str) and "_thermal.run" in item.code) == 1
    # q/p, cached force and normal RNG cache all roll back together.
    control.set_failure(-1)
    reference = make(potential=potential)
    reference.run(10000, stride=1000)
    assert e.state()["digest"] == reference.state()["digest"]


def test_gil_is_released_for_whole_trajectory():
    e = make(beads=8)
    start, stop = threading.Event(), threading.Event()
    count = [0]
    def worker():
        start.wait()
        while not stop.is_set():
            count[0] += 1
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        start.set()
        e.run(150000, stride=150000)
    finally:
        stop.set()
        thread.join()
    assert count[0] > 1000


def test_checkpoint_corruption_and_physics_rejected(tmp_path):
    e = make()
    e.run(10)
    state = e.state()
    state["q"][0, 0, 0] += 1
    before = e.state()
    with pytest.raises(ValueError, match="checksum"):
        e.restore(state)
    assert e.state()["digest"] == before["digest"]
    with pytest.raises(ValueError, match="physics"):
        make(dt=.03).restore(before)
    with pytest.raises(ValueError, match="physics"):
        make(thermostat="andersen").restore(before)


@pytest.mark.parametrize("changes", [
    {"thermostat": {"mode": "pile-l", "gamma": 1.}},
    {"thermostat": {"mode": "trpmd", "gamma_centroid": 1.}},
    {"thermostat": {"mode": "andersen", "interval": True}},
    {"thermostat": {"mode": "gle", "gle": {"A": [[-1.]]}}},
    {"thermostat": {"mode": "gle", "gle": {"A": [[1.]], "C": [[2.]]}}},
    {"thermostat": {"mode": "gle", "gle": {"A": [[1., 10.], [0., 1.]]}}},
    {"thermostat": {"mode": "piglet", "gle": {"A": [[1.]]}}},
    {"barostat": {"mode": "bzp", "pressure": 1.}},
    {"barostat": {"mode": "bzp"}, "cell": np.eye(3)},
    {"barostat": {"mode": "bzp", "pressure": 1.}, "cell": np.eye(3), "pbc": [True, True, False]},
    {"barostat": {"mode": "bzp", "pressure": 1.}, "cell": np.eye(3), "fixed_atoms": [1]},
    {"barostat": {"mode": "scr", "pressure": 1., "compressibility": 1.}, "cell": np.eye(3)},
    {"barostat": {"mode": "sc-isotropic"}},
    {"barostat": {"mode": "mttk", "pressure": 1.}, "cell": [[2, .2, 0], [0, 2, 0], [0, 0, 2]]},
    {"dt": float("nan")}, {"beta": 0}, {"integrator": "velocity-verlet"},
])
def test_invalid_configuration_is_explicit(changes):
    with pytest.raises((ValueError, NotImplementedError)):
        make(**changes)


@pytest.mark.parametrize("unit,factor", [("Pa", 1.), ("bar", 1.e5), ("atm", 101325.), ("MPa", 1.e6), ("GPa", 1.e9), ("kbar", 1.e8)])
def test_pressure_units_and_inverse_units(unit, factor):
    from pyrpmd.constants import BOHR_M, HARTREE_J
    from pyrpmd.units import parse_quantity
    expected = factor*BOHR_M**3/HARTREE_J
    assert parse_quantity("1 "+unit, "pressure") == pytest.approx(expected, rel=1.e-14)
    assert parse_quantity("1 "+unit+"^-1", "inverse_pressure") == pytest.approx(1/expected, rel=1.e-14)
    with pytest.raises(ValueError):
        parse_quantity("1 fs", "pressure")


def test_unit_aware_config_matches_atomic_api():
    from pyrpmd.native_ensemble import ensemble_from_config
    from pyrpmd.constants import AMU_TO_AU, ANGSTROM_TO_AU, FS_TO_AU, KB_AU
    from pyrpmd.units import parse_quantity
    config = dict(positions=[[0., 0., 0.]], geometry_unit="angstrom", masses=["1 amu"],
                  beads=3, temperature="300 K", timestep="0.1 fs", cell=(np.eye(3)*4).tolist(),
                  thermostat={"mode": "pile-l", "gamma_centroid": "0.01 fs^-1"},
                  barostat={"mode": "bzp", "pressure": "1 bar", "tau": "1 ps", "gamma": "0.002 fs^-1"}, seed=12)
    frozen = copy.deepcopy(config)
    a = ensemble_from_config(config)
    b = NativeEnsemble(np.zeros((3, 1, 3)), [AMU_TO_AU], 1/(KB_AU*300), .1*FS_TO_AU,
                       cell=np.eye(3)*4*ANGSTROM_TO_AU,
                       thermostat={"mode": "pile-l", "gamma_centroid": parse_quantity("0.01 fs^-1", "frequency")},
                       barostat={"mode": "bzp", "pressure": parse_quantity("1 bar", "pressure"),
                                 "tau": 1000*FS_TO_AU, "gamma": parse_quantity("0.002 fs^-1", "frequency")}, seed=12)
    assert a.fingerprint == b.fingerprint
    a.run(41); b.run(41)
    assert a.state()["digest"] == b.state()["digest"]
    assert config == frozen


def test_cli_streams_and_resumes_without_python_step_loop(tmp_path, capsys):
    import json
    from pyrpmd.native_ensemble import main, ensemble_from_config
    config = dict(schema_version="native-ensembles/1", positions=[[0., 0., 0.]], masses=[1.],
                  beads=3, beta=2., timestep=.02, seed=73, potential={"kind": "harmonic", "k": .5},
                  thermostat={"mode": "nm-gle", "gle": {"A": [[1., 2.], [-2., 1.]], "A_unit": "autime^-1"}},
                  run={"steps": 21, "stride": 4, "record_centroids": True})
    source=tmp_path/"input.json"; source.write_text(json.dumps(config))
    output=tmp_path/"out"
    assert main([str(source), "--output", str(output)]) == 0
    assert main([str(source), "--output", str(output), "--resume", str(output/"checkpoint.npz")]) == 0
    result=json.loads((output/"summary.json").read_text())
    assert result["step"] == 42 and result["python_step_callbacks"] == 0
    assert result["physical_converged"] is False
    parts=[np.load(path) for path in sorted(output.glob("trajectory-*.npy"))]
    expected=ensemble_from_config(config).run(42, stride=4, record_centroids=True)
    np.testing.assert_array_equal(np.concatenate(parts)[:, :18], expected["samples"])
    np.testing.assert_array_equal(np.concatenate(parts)[:, 18:].reshape(-1, 1, 3), expected["centroids"])
    capsys.readouterr()


def test_builtin_periodic_forces_and_cell_virial_are_consistent():
    cell=np.array([[2., 0, 0], [.3, 2.4, 0], [-.2, .1, 3.]])
    q=np.array([[[.2, .4, .6]]])
    def build(x, h):
        return NativeEnsemble(x, [1.], 2., .02, p=np.zeros_like(x), cell=h,
                              thermostat="none", potential={"kind": "periodic_cosine", "heights": [.2, .3, .4]})
    reference=build(q,cell).state(); eps=1.e-6
    for j in range(3):
        delta=np.zeros_like(q);delta[0,0,j]=eps
        difference=(build(q+delta,cell).state()["energy"]-build(q-delta,cell).state()["energy"])/(2*eps)
        np.testing.assert_allclose(-difference, reference["force"][0,0,j], atol=2.e-10)
    # Affine cell+position changes leave fractional coordinates unchanged.
    for i in range(3):
        for j in range(3):
            deformation=np.zeros((3,3));deformation[i,j]=eps
            plus=np.eye(3)+deformation;minus=np.eye(3)-deformation
            value=(build(q@plus.T,cell@plus.T).state()["energy"]-build(q@minus.T,cell@minus.T).state()["energy"])/(2*eps)
            np.testing.assert_allclose(-value, reference["virial"][0,i,j], atol=2.e-10)


def test_unused_potential_fields_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        make(potential={"kind": "free", "k": .1})
