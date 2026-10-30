"""Native block integration, not an isolated microbenchmark.

The small compiled fixture supplies a Fortran-compatible BIND(C) signature.
The real author-derived Fortran BKMP2 tests are in the GitHub integration study.
"""
from pathlib import Path
import ctypes
import shutil
import subprocess
import threading

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.cv import Distance, BondDifference, Cartesian, COMDistance, CVStack, RPMDrateCoordinate
from pyrpmd.bias import UmbrellaBias, FluxMetricBias, CompositeBias
from pyrpmd.potentials import HarmonicPotential, PythonPotential, CABIPotential
from pyrpmd.potentials.native import NativePESBinding
from pyrpmd.propagation import PropagationError, scientific_config
from pyrpmd.workflows.rate import project_positions, project_momenta

pytestmark = pytest.mark.skipif(not native_capabilities()["available"], reason="native build required")


@pytest.fixture(scope="module")
def native_library(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("C compiler required")
    root = tmp_path_factory.mktemp("native-pes")
    source = root / "pes.c"
    source.write_text(r'''
#include <stdint.h>
#include <math.h>
static int calls=0, fail_at=-1, nonfinite=0;
void reset(int fail, int nan) {calls=0;fail_at=fail;nonfinite=nan;}
int get_calls(void) {return calls;}
void batch(int32_t B, const double* q, double* e, double* f, int32_t* status) {
  ++calls;
  for(int32_t b=0;b<B;++b){
    e[b]=0;status[b]=(calls==fail_at && !nonfinite)?7:0;
    for(int j=0;j<9;++j){int i=9*b+j;e[b]+=.005*q[i]*q[i];f[i]=-.01*q[i];}
    if(calls==fail_at && nonfinite)e[b]=NAN;
  }
}
''')
    library = root / "pes.so"
    subprocess.run([cc, "-O2", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    return library


class CompiledPotential:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library))
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.fn = self.lib.batch
        self.fn.argtypes = [ctypes.c_int32, pointer, pointer, pointer,
                            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")]
        self.fn.restype = None
        self.lib.reset.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.reset.restype = None
        self.lib.get_calls.argtypes = []
        self.lib.get_calls.restype = ctypes.c_int
        self.lock = threading.RLock()
        self.closed = False
        self.lib.reset(-1, 0)

    def evaluate(self, q):
        e = np.empty(len(q));f = np.empty_like(q);status = np.empty(len(q), dtype=np.int32)
        with self.lock:
            if self.closed:
                raise RuntimeError("closed")
            self.fn(len(q), q, e, f, status)
        return e, f, status

    def native_binding(self):
        return NativePESBinding("fortran-batch-v1", self.fn, 3, self, self.lock,
                                is_open=lambda: not self.closed)

    def close(self):
        with self.lock:
            self.closed = True


def state(beads=8):
    rng = np.random.default_rng(149)
    q = np.repeat(np.array([[-2., .1, 0], [0, 0, 0], [1.7, .2, .1]])[None], beads, axis=0)
    q += rng.normal(scale=.01, size=q.shape)
    return State(q, rng.normal(size=q.shape))


def dynamics(provider, beads=8, **kwargs):
    return Dynamics(provider, [1800., 1800., 1800.], 220., beads, .2, seed=173, **kwargs)


def coordinate():
    return RPMDrateCoordinate([[0], [1, 2]], np.ones(3)*1800,
                             np.array([[-1.75,0,0], [0,0,0], [1.75,0,0]]),
                             [[0,1]], [[1,2]], 10, equivalent_paths=[[0,2,1]])


@pytest.mark.parametrize("beads", [1, 2, 3, 8, 16, 32])
def test_full_states_observations_and_independent_fft(native_library, beads):
    pes = CompiledPotential(native_library)
    dyn = dynamics(pes, beads)
    cv = CVStack([Distance([0, 1]), Distance([1, 2], estimator="bead_average"),
                  BondDifference([0,1,2]), COMDistance([[0],[1,2]], dyn.masses), Cartesian(1, "y")])
    initial = state(beads);a, b = initial.copy(), initial.copy()
    got = dyn.propagate_block(a, 17, cv=cv, record_states=True, backend="native")
    expected = dyn.propagate_block(b, 17, cv=cv, record_states=True, backend="python")
    np.testing.assert_allclose(got.q, expected.q, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(got.p, expected.p, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(got.values, expected.values, atol=1e-13)
    np.testing.assert_allclose(got.energies, expected.energies, atol=1e-13)
    np.testing.assert_array_equal(got.steps, np.arange(1,18))
    np.testing.assert_array_equal(got.times, expected.times)
    fft = dynamics(pes, beads, native=False)
    c = initial.copy()
    for _ in range(17):
        fft.step(c)
    np.testing.assert_allclose(a.q, c.q, atol=2e-11)
    np.testing.assert_allclose(a.p, c.p, atol=2e-10)
    assert got.backend == "native"


@pytest.mark.parametrize("kind", ["umbrella", "flux", "constraint", "fragments"])
def test_bias_constraints_and_fragment_projection(native_library, kind):
    pes = CompiledPotential(native_library);dyn = dynamics(pes)
    s = state();cv = coordinate();kw = {"cv": cv, "record_states": True}
    if kind == "umbrella":
        cv2 = CVStack([Distance([0,1]), Distance([1,2], estimator="bead_average")])
        kw.update(cv=cv2, bias=UmbrellaBias(cv2, [2., 1.7], [[.03,.005],[.005,.04]]))
    elif kind == "flux":
        kw["bias"] = CompositeBias(UmbrellaBias(cv, [0.9], .02), FluxMetricBias(cv, dyn.masses, dyn.beta))
    elif kind == "constraint":
        surface = cv.dividing_surface(1.001)
        s.q[:] = project_positions(s.q, surface, dyn.masses)
        project_momenta(s.p, s.q, surface, dyn.masses)
        kw.update(constraint=surface, cv=surface)
    else:
        kw["fragment_data"] = dyn._fragment_data(s, [[0], [1,2]])
        dyn._fix_fragments(s, kw["fragment_data"])
    a, b = s.copy(), s.copy()
    x = dyn.propagate_block(a, 25, backend="native", **kw)
    y = dyn.propagate_block(b, 25, backend="python", **kw)
    np.testing.assert_allclose(x.q, y.q, atol=3e-11)
    np.testing.assert_allclose(x.p, y.p, atol=3e-10)
    np.testing.assert_allclose(x.values, y.values, atol=3e-12)
    np.testing.assert_allclose(x.energies, y.energies, atol=3e-12)
    if kind == "constraint":
        assert np.max(abs(x.values)) < 1e-10


def test_native_chunking_is_exact_and_reversible(native_library):
    dyn = dynamics(CompiledPotential(native_library));s = state();a, b = s.copy(), s.copy()
    x = dyn.propagate_block(a, 33, backend="native")
    partial = [dyn.propagate_block(b, n, backend="native") for n in [1,7,3,22]]
    np.testing.assert_array_equal(a.q, b.q)
    np.testing.assert_array_equal(a.p, b.p)
    np.testing.assert_array_equal(x.energies, np.concatenate([p.energies for p in partial]))
    assert a.step == b.step and a.time == b.time
    a.p *= -1
    dyn.propagate_block(a, 33, backend="native")
    np.testing.assert_allclose(a.q, s.q, atol=3e-11)
    np.testing.assert_allclose(a.p, -s.p, atol=3e-10)


def test_andersen_rng_schedule_and_fragments(native_library):
    a, b = state(), state();pn, pp = CompiledPotential(native_library), CompiledPotential(native_library)
    dn = dynamics(pn, loop_backend="native", block_steps=4)
    dp = dynamics(pp, loop_backend="python", block_steps=7)
    dn.thermalize(a, 35, interval=6, fragments=[[0], [1,2]])
    dp.thermalize(b, 35, interval=6, fragments=[[0], [1,2]])
    np.testing.assert_allclose(a.q,b.q,atol=3e-12)
    np.testing.assert_allclose(a.p,b.p,atol=3e-11)
    assert dn.rng.bit_generator.state == dp.rng.bit_generator.state
    assert dn.loop_stats["native_steps"] == 35


@pytest.mark.parametrize("nonfinite", [0,1])
def test_failed_step_rolls_back_only_the_invalid_step_and_never_retries(native_library, nonfinite):
    pes = CompiledPotential(native_library);dyn = dynamics(pes);s = state();reference = s.copy()
    dyn.propagate_block(reference, 2, record_energy=False, backend="native")
    pes.lib.reset(6, nonfinite)  # second force of step 3
    with pytest.raises(PropagationError) as caught:
        dyn.propagate_block(s, 10, record_energy=False, backend="native", record_states=True)
    result = caught.value.result
    assert result.completed == 2 and len(result.steps) == 2
    assert pes.lib.get_calls() == 6  # no Python retry and no speculative later force
    np.testing.assert_array_equal(s.q, reference.q)
    np.testing.assert_array_equal(s.p, reference.p)
    assert s.step == 2 and s.time == reference.time


def test_masks_empty_blocks_and_input_validation(native_library):
    dyn = dynamics(CompiledPotential(native_library));s = state()
    result = dyn.propagate_block(s, 6, observe=[0,1,0,0,1,0], backend="native")
    np.testing.assert_array_equal(result.steps,[2,5]);assert s.step == 6
    assert len(dyn.propagate_block(s,0).steps)==0
    for bad in (-1,True,1.1,4097):
        with pytest.raises(ValueError):dyn.propagate_block(s,bad)
    with pytest.raises(ValueError):dyn.propagate_block(s,2,observe=[1,2])
    with pytest.raises(ValueError):dyn.propagate_block(s,2,backend="typo")
    with pytest.raises(ValueError):dyn.propagate_block(s,2,constraint_tolerance=0)


def test_python_custom_pes_cv_bias_and_identity_forwarding(native_library):
    class Shifted(Distance):
        calls = 0
        def centroid_value_gradient(self,q):
            self.calls += 1
            value,g = super().centroid_value_gradient(q)
            return value - .3,g
    dyn = dynamics(CompiledPotential(native_library));custom = Shifted([0,1]);a,b = state(),state()
    auto = dyn.propagate_block(a,5,cv=custom,observe=[0,1,0,0,1])
    assert custom.calls == 2 and auto.backend == "python"
    ref = dyn.propagate_block(b,5,cv=custom,observe=[0,1,0,0,1],backend="python")
    np.testing.assert_array_equal(a.q,b.q);np.testing.assert_array_equal(auto.values,ref.values)
    with pytest.raises(RuntimeError,match="required but unsupported"):
        dyn.propagate_block(a,1,cv=custom,backend="native")
    # A native PES wrapped with unit conversion MUST NOT bypass that conversion.
    delegated = PythonPotential(dyn.potential)
    assert dynamics(delegated).propagate_block(state(),2).backend == "native"
    converted = PythonPotential(dyn.potential,units={"positions":"angstrom","energy":"eV"})
    assert dynamics(converted).propagate_block(state(),2).backend == "python"
    assert dynamics(HarmonicPotential()).propagate_block(state(),2).backend == "python"
    custom_bias = UmbrellaBias(custom, [1.7], .01)
    assert dyn.propagate_block(state(),2,bias=custom_bias).backend == "python"


def test_closed_handle_and_execution_identity(native_library):
    pes = CompiledPotential(native_library);dyn=dynamics(pes);s=state();before=s.copy()
    pes.close()
    with pytest.raises(RuntimeError,match="closed"):
        dyn.propagate_block(s,3,backend="native")
    np.testing.assert_array_equal(s.q,before.q);assert s.step == 0
    assert scientific_config({"dynamics":{"loop_backend":"native","block_steps":64},"beads":8}) == {"beads":8}
    assert scientific_config({"dynamics":{"native":True,"loop_backend":"python"}}) == {"dynamics":{"native":True}}


def test_real_cabi_context_lifetime(tmp_path):
    cc=shutil.which("cc") or shutil.which("gcc")
    if cc is None:pytest.skip("C compiler required")
    root=Path(__file__).resolve().parents[1];lib=tmp_path/"harmonic.so"
    subprocess.run([cc,"-shared","-fPIC","-I",str(root/"native/include"),str(root/"native/examples/harmonic_pes.c"),"-o",str(lib)],check=True)
    pes=CABIPotential(lib)
    dyn=Dynamics(pes,[1800.],220.,8,.1)
    s=State(np.ones((8,1,3)),np.ones((8,1,3)))
    assert dyn.propagate_block(s,9).backend=="native"
    binding=pes.native_binding();finished=threading.Event()
    def close():pes.close();finished.set()
    with binding.acquire():
        t=threading.Thread(target=close);t.start()
        assert not finished.wait(.02)
    t.join(timeout=2);assert finished.is_set()
    with pytest.raises(RuntimeError,match="closed"):dyn.propagate_block(s,1)


def _native_collision_config():
    return {
        "beads": 8,
        "dt_au": .2,
        "collision": {"max_steps": 12},
        "outcomes": {"check_interval": 4},
        "output": {"trajectory_stride": 1, "checkpoint_steps": 12},
    }


def test_drpmd_native_failure_keeps_prefix_without_replaying_pes(native_library, tmp_path):
    from pyrpmd.workflows.drpmd import _trajectory, _load_npz

    pes = CompiledPotential(native_library)
    initial = state()
    reference = initial.copy()
    dynamics(pes).propagate_block(reference, 2, backend="native")
    dyn = dynamics(pes, loop_backend="native", block_steps=12)
    # One initial-energy call precedes the two force calls per step.
    # Fail the second force of step 3, after exactly two valid steps.
    pes.lib.reset(7, 0)
    result = _trajectory(dyn, initial.copy(), None, _native_collision_config(),
                         tmp_path, 0, 0, {}, "native-prefix-failure", False)

    assert result["status"] == "failed"
    assert "native PES frame" in result["reason"]
    assert result["steps"] == 2
    assert result["propagation"] == {"native_steps": 2, "python_steps": 0}
    assert pes.lib.get_calls() == 7  # no force retry, energy retry or later step
    with np.load(tmp_path / "trajectories/trajectory-00000000.npz") as saved:
        np.testing.assert_array_equal(saved["final_q"], reference.q)
        np.testing.assert_array_equal(saved["final_p"], reference.p)
        np.testing.assert_array_equal(saved["steps"], [0, 1, 2])
    metadata, checkpoint = _load_npz(tmp_path / "trajectories/trajectory-00000000.checkpoint.npz")
    assert metadata["done"] == metadata["step"] == 2
    assert metadata["propagation"] == result["propagation"]
    np.testing.assert_array_equal(checkpoint["q"], reference.q)
    np.testing.assert_array_equal(checkpoint["p"], reference.p)


def test_drpmd_python_classifier_boundaries_keep_native_prefix_and_counts(native_library, tmp_path):
    from pyrpmd.workflows.drpmd import _trajectory, _load_npz

    pes = CompiledPotential(native_library)
    initial = state()
    expected = dynamics(pes).propagate_block(initial.copy(), 8, backend="native", record_states=True)
    dyn = dynamics(pes, loop_backend="native", block_steps=12)
    launched = initial.copy()
    seen = []

    def classifier(current):
        # Native dynamics must call an arbitrary Python classifier at exactly
        # its configured boundaries with the live, complete State object.
        assert current is launched
        seen.append(current.step)
        np.testing.assert_array_equal(current.q, expected.q[current.step - 1])
        np.testing.assert_array_equal(current.p, expected.p[current.step - 1])
        assert current.time == expected.times[current.step - 1]
        if current.step == 8:
            raise RuntimeError("classifier rejected second boundary")
        return None

    pes.lib.reset(-1, 0)
    result = _trajectory(dyn, launched, classifier, _native_collision_config(),
                         tmp_path, 0, 0, {}, "native-classifier-boundaries", False)

    assert seen == [4, 8]
    assert result["status"] == "failed"
    assert result["reason"] == "RuntimeError: classifier rejected second boundary"
    assert result["steps"] == 7
    assert result["propagation"] == {"native_steps": 7, "python_steps": 0}
    assert dyn.propagation_stats["native_steps"] == 7
    assert pes.lib.get_calls() == 17  # initial energy + 8 attempted physical steps
    with np.load(tmp_path / "trajectories/trajectory-00000000.npz") as saved:
        np.testing.assert_array_equal(saved["final_q"], expected.q[6])
        np.testing.assert_array_equal(saved["final_p"], expected.p[6])
        np.testing.assert_array_equal(saved["steps"], np.arange(8))
    metadata, checkpoint = _load_npz(tmp_path / "trajectories/trajectory-00000000.checkpoint.npz")
    assert metadata["done"] == metadata["step"] == 7
    assert metadata["propagation"] == result["propagation"]
    np.testing.assert_array_equal(checkpoint["q"], expected.q[6])
    np.testing.assert_array_equal(checkpoint["p"], expected.p[6])
