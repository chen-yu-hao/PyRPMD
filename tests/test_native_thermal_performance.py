"""Exact stochastic contracts and binary propagation transactions."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("pyrpmd._thermal")
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.native_ensemble import NativeEnsemble
from pyrpmd.potentials import HarmonicPotential


def bath(mode, beads, rng="philox-v1"):
    result = {"mode": mode, "rng": rng}
    if "gle" in mode or mode == "piglet":
        result["gle"] = {"A": [[1., .2], [-.2, 1.]]}
    if mode == "piglet":
        result["gle"].update(A=[result["gle"]["A"]]*beads, C=np.eye(2).tolist(),
                             beta=2., beads=beads, provenance="Synthetic canonical test covariance")
    return result


def ensemble(mode="pile-l", *, beads=8, atoms=9, rng="philox-v1", seed=49):
    options = {"potential": {"kind": "harmonic", "k": .5}}
    if mode in ("bzp", "mttk", "rgb", "scr"):
        barostat = {"mode": mode, "tau": 2.}
        if mode == "rgb":
            barostat.update(stress=(np.eye(3)*.1).tolist(), reference_cell=(np.eye(3)*3).tolist())
        else:
            barostat["pressure"] = .1
        if mode == "scr":
            barostat["compressibility"] = .1
        mode = "pile-l"
        options.update(potential={"kind": "periodic_cosine", "heights": [.1, .2, .3]},
                       cell=np.eye(3)*3, barostat=barostat)
    return NativeEnsemble(np.full((beads, atoms, 3), .1), np.ones(atoms), 2., .01,
                          thermostat=bath(mode, beads, rng), seed=seed, **options)


@pytest.mark.parametrize("mode", ["pile-l", "pile-g", "andersen", "langevin", "svr", "ffl", "nhc", "trpmd", "gle", "nm-gle", "nm-gle-g", "piglet", "bzp"])
@pytest.mark.parametrize("beads", [1, 7, 32])
def test_philox_checkpoint_owns_seed_counter_and_auxiliaries(tmp_path, mode, beads):
    whole = ensemble(mode, beads=beads)
    split = ensemble(mode, beads=beads)
    whole.run(19)
    split.run(7)
    split.save(tmp_path/"checkpoint.npz")
    resumed = ensemble(mode, beads=beads, seed=1234567)
    resumed.load(tmp_path/"checkpoint.npz")
    resumed.run(12)
    assert resumed.state()["rng"].startswith("philox-v1\n")
    assert resumed.state()["digest"] == whole.state()["digest"]


@pytest.mark.parametrize("mode,beads", [("mttk", 7), ("rgb", 7), ("scr", 1)])
def test_other_philox_pistons_replay_complete_state(mode, beads):
    engine = ensemble(mode, beads=beads)
    engine.run(7)
    saved = engine.state()
    engine.run(11)
    resumed = ensemble(mode, beads=beads, seed=381)
    resumed.restore(saved)
    resumed.run(11)
    assert resumed.state()["digest"] == engine.state()["digest"]


def test_explicit_legacy_is_the_default_and_rejects_philox_checkpoint():
    q = np.full((8, 9, 3), .1)
    default = NativeEnsemble(q, np.ones(9), 2., .01, seed=49,
                             potential={"kind": "harmonic", "k": .5}, thermostat="pile-l")
    explicit = ensemble(rng="mt19937_64")
    default.run(11); explicit.run(11)
    assert default.state()["digest"] == explicit.state()["digest"]
    with pytest.raises(ValueError, match="physics"):
        default.restore(ensemble().state())


@pytest.mark.parametrize("backend", ["native", "python"])
def test_binary_token_restores_failure_and_retry_without_checkpoint_conversion(backend):
    dynamics = Dynamics(HarmonicPotential(k=.5), [1.]*9, 2., 8, .01,
                        thermostat=bath("pile-l", 8), seed=19, loop_backend=backend)
    state = State(np.full((8, 9, 3), .1), np.zeros((8, 9, 3)))
    dynamics.propagate(state, 3)
    token = dynamics._operator_rollback_token()
    saved = dynamics.operator_state_dict()
    dynamics.propagate(state, 5)
    dynamics._restore_operator_rollback_token(token)
    assert saved == dynamics.operator_state_dict()



def test_overlapping_tokens_are_owned_and_immutable():
    engine = ensemble()
    first = engine._rollback_token()
    before = engine.state()
    engine.run(3)
    second = engine._rollback_token()
    middle = engine.state()
    engine.run(4)
    engine._restore_rollback_token(first)
    assert engine.state()["digest"] == before["digest"]
    engine._restore_rollback_token(second)
    assert engine.state()["digest"] == middle["digest"]
    other = ensemble()
    with pytest.raises(ValueError, match="another native ensemble"):
        other._restore_rollback_token(first)


@pytest.mark.parametrize("beads", [1, 7, 32])
def test_initial_energy_retains_numpy_reduction_order(beads):
    rng = np.random.default_rng(7149)
    q, p = rng.normal(size=(2, beads, 13, 3))
    masses = np.linspace(.7, 3.1, 13)
    engine = NativeEnsemble(q, masses, 2.3, .01, p=p, potential={"kind": "harmonic", "k": .7})
    before = engine.state()
    difference = q - np.roll(q, -1, axis=0)
    expected = float(np.sum(before["energy"]) + np.sum(p*p/(2*masses[None, :, None]))
                     + .5*(beads/2.3)**2*np.sum(masses[None, :, None]*difference*difference))
    assert engine._initial_energy(q, p, masses, 2.3).hex() == expected.hex()
    assert engine.state()["digest"] == before["digest"]

def test_rollback_does_not_call_public_json_checkpoint(monkeypatch):
    dynamics = Dynamics(HarmonicPotential(k=.5), [1.]*9, 2., 8, .01,
                        thermostat=bath("pile-l", 8), seed=19)
    state = State(np.full((8, 9, 3), .1), np.zeros((8, 9, 3)))
    dynamics.propagate(state, 3)
    saved = dynamics.operator_state_dict()
    original = state.copy()
    def forbidden():
        raise AssertionError("rollback used the JSON checkpoint interface")
    monkeypatch.setattr(dynamics, "operator_state_dict", forbidden)
    class FailingObservation:
        dimension = 1
        calls = 0
        def value(self, coordinates):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("observation failure after completed steps")
            return np.array([coordinates.mean()])
    with pytest.raises(RuntimeError, match="observation failure"):
        dynamics.propagate(state, 5, observe=FailingObservation())
    assert Dynamics.operator_state_dict(dynamics) == saved
    np.testing.assert_array_equal(state.q, original.q)
    np.testing.assert_array_equal(state.p, original.p)


def test_philox_algorithm_known_answers(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path/"vectors.cpp"
    source.write_text('''#include "philox.h"
#include <iostream>
int main(){for(auto x:pyrpmd::philox::generate({0,0,0,0},{0,0}))std::cout<<std::hex<<x<<" ";
for(auto x:pyrpmd::philox::generate({0xffffffff,0xffffffff,0xffffffff,0xffffffff},{0xffffffff,0xffffffff}))std::cout<<std::hex<<x<<" ";}
''')
    subprocess.run([compiler, "-std=c++17", "-I", str(Path(__file__).resolve().parents[1]/"native"),
                    str(source), "-o", str(tmp_path/"vectors")], check=True, capture_output=True)
    result = subprocess.check_output([str(tmp_path/"vectors")], text=True).split()
    assert result == ["6627e8d5", "e169c58d", "bc57ac4c", "9b00dbd8",
                      "408f276d", "41c83b0e", "a20bc7c6", "6d5451fd"]


@pytest.mark.parametrize("rng", ["mt19937_64", "philox-v1"])
def test_threads_are_bitwise_identical(tmp_path, rng):
    results = []
    for threads in (1, 2, 4, 8):
        output = tmp_path/f"{threads}.npz"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", str(output), rng],
                       env={**os.environ, "OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": "1"},
                       check=True, capture_output=True)
        with np.load(output) as data:
            results.append({key: data[key].copy() for key in data.files})
    for candidate in results[1:]:
        for key in results[0]:
            np.testing.assert_array_equal(candidate[key], results[0][key], err_msg=key)


def worker(path, rng):
    result = {}
    for mode in ("pile-l", "pile-g", "gle", "piglet", "nhc", "bzp"):
        engine = ensemble(mode, beads=32, atoms=384, rng=rng)
        engine.run(5)
        for key, value in engine.state().items():
            result[mode+"_"+key] = value
    np.savez(path, **result)


def workflow_config():
    return {"task": "pimd", "temperature": "300 K", "beads": 4, "seed": 45,
            "system": {"symbols": ["H", "H"], "geometry": [[0., 0., 0.], [.1, .2, .3]], "geometry_unit": "bohr"},
            "potential": {"provider": "builtin", "name": "harmonic", "params": {"k": .01}},
            "dynamics": {"timestep": "0.1 autime", "loop_backend": "native"},
            "sampling": {"engine": "native", "thermostat": "pile-l", "equilibration_steps": 7,
                         "production_steps": 30, "sample_interval": 3,
                         "thermostat_params": {"rng": "philox-v1"}},
            "output": {"checkpoint_steps": 11}}


def test_real_input_workflow_and_checkpoint_cadence(tmp_path):
    from pyrpmd.runner import run, resume_run
    from pyrpmd.input import JobSpec
    input_path = tmp_path/"input.inp"
    text = """task = "pimd"
temperature = "300 K"
beads = 4
seed = 45
[system]
symbols = ["H", "H"]
geometry = [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]
geometry_unit = "bohr"
[potential]
provider = "builtin"
name = "harmonic"
[potential.params]
k = 0.01
[dynamics]
timestep = "0.1 autime"
loop_backend = "native"
[sampling]
engine = "native"
thermostat = "pile-l"
equilibration_steps = 7
production_steps = 30
sample_interval = 3
[sampling.thermostat_params]
rng = "philox-v1"
[output]
checkpoint_steps = 11
"""
    input_path.write_text(text)
    first = run(JobSpec.from_file(input_path), output=tmp_path/"first")
    input_path.write_text(text.replace("checkpoint_steps = 11", "checkpoint_steps = 3"))
    second = run(JobSpec.from_file(input_path), output=tmp_path/"second")
    assert first["python_step_callbacks"] == second["python_step_callbacks"] == 0
    assert first["estimators"] == second["estimators"]
    assert resume_run(tmp_path/"second")["estimators"] == first["estimators"]
    resolved = json.loads((tmp_path/"first"/"input"/"resolved.json").read_text())
    assert resolved["sampling"]["thermostat_params"]["rng"] == "philox-v1"


@pytest.mark.parametrize("rng", [None, [], 3, "philox", "unknown"])
def test_invalid_rng_rejected_before_output(tmp_path, rng):
    from pyrpmd.runner import run
    config = workflow_config()
    config["sampling"]["thermostat_params"]["rng"] = rng
    with pytest.raises(ValueError, match="rng"):
        run(config, output=tmp_path/"output")
    assert not (tmp_path/"output").exists()


if __name__ == "__main__" and sys.argv[1] == "worker":
    worker(sys.argv[2], sys.argv[3])


def test_stale_extension_cannot_silently_select_legacy_noise(monkeypatch):
    from pyrpmd import _thermal
    monkeypatch.setattr(_thermal, "THERMAL_KERNEL_API", 0)
    with pytest.raises(RuntimeError, match="rebuild PyRPMD"):
        ensemble()
