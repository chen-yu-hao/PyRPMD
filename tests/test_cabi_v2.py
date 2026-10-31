"""Real compiled v2 PES tests: batch/cell layouts, forces and strain virials."""

import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials import CABIPotential, PotentialError, make_potential
from pyrpmd.potentials.base import validate_result
from pyrpmd.propagation import NativeUnavailableError


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native/examples/periodic_cosine_pes.c"
CELL = np.array([[6., 0.2, 0.1], [1.1, 5., 0.3], [.4, .7, 4.]])
PBC = np.array([True, False, True])
Q = np.array([[[.2, .4, .8]], [[-.6, .3, 1.1]], [[1.7, .2, -.4]]])


def _compile(directory, name, text=None, *, v1=False):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler required for ABI tests")
    source = SOURCE
    if text is not None:
        source = directory / (name + ".c")
        source.write_text(text)
    output = directory / (name + ".so")
    sources = [str(source)]
    if v1:
        sources.append(str(ROOT / "native/examples/harmonic_pes.c"))
    subprocess.run([
        compiler, "-std=c11", "-O2", "-fPIC", "-shared", "-I",
        str(ROOT / "native/include"), *sources, "-lm", "-o", str(output),
    ], check=True, capture_output=True, text=True)
    return output


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return _compile(tmp_path_factory.mktemp("cabi-v2"), "periodic", v1=True)


def _system(cell=CELL, pbc=PBC):
    return {"symbols": ["H"], "atomic_numbers": [1], "charge": 0,
            "multiplicity": 1, "cell": cell, "pbc": pbc}


@pytest.mark.parametrize("pbc", [[True, True, True], PBC, [False, False, False]])
def test_batched_force_and_full_affine_strain_virial(library, pbc):
    cells = np.stack([CELL, CELL * 1.2, CELL * .8])
    with CABIPotential(library) as potential:
        result = potential.evaluate(Q, cell=cells, pbc=pbc)
        epsilon = 1e-5
        numerical_force = np.zeros_like(Q)
        numerical_virial = np.zeros_like(result.virial)
        for j in range(3):
            delta = np.zeros_like(Q)
            delta[..., j] = epsilon
            numerical_force[..., j] = -(
                potential.evaluate(Q + delta, cell=cells, pbc=pbc).energy
                - potential.evaluate(Q - delta, cell=cells, pbc=pbc).energy
            )[:, None] / (2 * epsilon)
        for a in range(3):
            for b in range(3):
                strain = np.zeros((3, 3))
                strain[a, b] = epsilon
                plus, minus = np.eye(3) + strain, np.eye(3) - strain
                numerical_virial[:, a, b] = -(
                    potential.evaluate(Q @ plus.T, cell=cells @ plus.T, pbc=pbc).energy
                    - potential.evaluate(Q @ minus.T, cell=cells @ minus.T, pbc=pbc).energy
                ) / (2 * epsilon)
        np.testing.assert_allclose(result.forces, numerical_force, atol=1e-10, rtol=1e-9)
        np.testing.assert_allclose(result.virial, numerical_virial, atol=1e-10, rtol=1e-8)
        # This potential has explicit cell dependence: a position-only virial
        # would silently report a different pressure even with correct forces.
        naive = np.einsum("bni,bnj->bij", result.forces, Q)
        assert not np.allclose(result.virial, naive)


def test_one_compiled_call_preserves_batch_order_and_partial_pbc(library):
    with CABIPotential(library, system=_system()) as potential:
        compiled = potential._evaluate
        calls = []

        def observed(context, inputs, outputs, error):
            calls.append(inputs._obj.nstructures)
            return compiled(context, inputs, outputs, error)

        potential._evaluate = observed
        result = potential.evaluate(Q)
        assert calls == [len(Q)]
        np.testing.assert_allclose(
            result.energy,
            [potential.evaluate(frame[None]).energy[0] for frame in Q],
        )
        for axis in (0, 2):
            shifted = potential.evaluate(Q + 3 * CELL[axis])
            np.testing.assert_allclose(result.energy, shifted.energy, atol=3e-14)
            np.testing.assert_allclose(result.forces, shifted.forces, atol=3e-14)
            np.testing.assert_allclose(result.virial, shifted.virial, atol=3e-14)
        assert not np.allclose(result.energy, potential.evaluate(Q + CELL[1]).energy)


def test_per_frame_cells_and_flags_match_scalar_and_do_not_rebind(library):
    cells = np.stack([CELL, CELL * 1.1, CELL * .9])
    flags = np.array([[True, False, True], [True, True, True], [False, False, False]])
    with CABIPotential(library, system=_system()) as potential:
        before = potential.evaluate(Q)
        batch = potential.evaluate(Q, cell=cells, pbc=flags)
        for i in range(len(Q)):
            scalar = potential.evaluate(Q[i:i+1], cell=cells[i], pbc=flags[i])
            np.testing.assert_array_equal(batch.energy[i:i+1], scalar.energy)
            np.testing.assert_array_equal(batch.forces[i:i+1], scalar.forces)
            np.testing.assert_array_equal(batch.virial[i:i+1], scalar.virial)
        np.testing.assert_array_equal(before.energy, potential.evaluate(Q).energy)
        potential.bind_system(_system(CELL * 1.2))
        assert not np.allclose(before.energy, potential.evaluate(Q).energy)
        # Returning owned arrays keeps prior forces/virials intact across calls.
        np.testing.assert_array_equal(before.energy,
                                      potential.evaluate(Q, cell=CELL, pbc=PBC).energy)


def test_isolated_v2_harmonic_and_explicit_cell_reset(library):
    with CABIPotential(library, system=_system()) as potential:
        result = potential.evaluate(Q, cell=None)
        np.testing.assert_allclose(result.energy, .5 * np.sum(Q * Q, axis=(1, 2)))
        np.testing.assert_array_equal(result.forces, -Q)
        np.testing.assert_array_equal(result.virial, -np.einsum("bni,bnj->bij", Q, Q))
        assert potential.cell is not None  # Per-call reset is not persistent.


def test_factory_and_explicit_version_selection(library):
    with make_potential({"provider": "cabi", "library": str(library)}, _system()) as potential:
        assert potential.abi_version == 2
        assert potential.metadata["supports_pbc"] is True
        assert potential.evaluate(Q).virial is not None
    with CABIPotential(library, abi_version=1) as potential:
        assert potential.abi_version == 1
        np.testing.assert_array_equal(potential.evaluate(Q).forces, -Q)
        assert potential.evaluate(Q).virial is None
        with pytest.raises(PotentialError, match="require.*v2"):
            potential.evaluate(Q, cell=CELL)
    with pytest.raises(PotentialError, match="v1 is nonperiodic"):
        CABIPotential(library, system=_system(), abi_version=1)


@pytest.mark.parametrize("bad_version", [0, 3, True, "2", 2.0])
def test_invalid_version_is_rejected_before_loading(bad_version):
    with pytest.raises(ValueError, match="abi_version"):
        CABIPotential("not-a-library.so", abi_version=bad_version)


@pytest.mark.parametrize("cell,pbc", [
    (CELL, [1, 0, 1]),
    (np.zeros((3, 3)), PBC),
    (np.full((3, 3), np.nan), PBC),
    (np.zeros((2, 3, 3)), PBC),
    (CELL, np.ones((2, 3), dtype=bool)),
    (None, [True, False, False]),
])
def test_invalid_geometry_never_reaches_compiled_code(library, cell, pbc):
    with CABIPotential(library) as potential:
        def unexpected(*args):
            pytest.fail("invalid geometry reached compiled evaluator")
        potential._evaluate = unexpected
        with pytest.raises(ValueError):
            potential.evaluate(Q, cell=cell, pbc=pbc)


def test_failed_rebind_keeps_original_geometry_and_rejects_closed(library):
    with CABIPotential(library, system=_system()) as potential:
        reference = potential.evaluate(Q).energy
        for bad in ({"atomic_numbers": [8]}, {"charge": 1}, {"symbols": ["H", "H"]}):
            with pytest.raises(PotentialError, match="mismatch|disagree"):
                potential.bind_system({**_system(), **bad})
            np.testing.assert_array_equal(potential.evaluate(Q).energy, reference)
    potential.close()
    with pytest.raises(PotentialError, match="closed"):
        potential.evaluate(Q)
    with pytest.raises(PotentialError, match="closed"):
        potential.bind_system(_system())


def test_dict_system_cannot_change_species_or_override_identity(library):
    # Worker serialization often omits the computed atomic_numbers property.
    system = {"symbols": ["D"], "cell": CELL, "pbc": PBC}
    with CABIPotential(library, system=system) as potential:
        reference = potential.evaluate(Q).energy
        with pytest.raises(PotentialError, match="element ordering mismatch"):
            potential.bind_system({**system, "symbols": ["O"]})
        np.testing.assert_array_equal(potential.evaluate(Q).energy, reference)
        potential.bind_system({**system, "symbols": ["T"]})
        np.testing.assert_array_equal(potential.evaluate(Q).energy, reference)
    with pytest.raises(PotentialError, match="element ordering mismatch"):
        CABIPotential(library, system={"symbols": ["O"]})
    for config in ({"symbols": ["O"]}, {"atomic_numbers": [8]}, {"natoms": 2}):
        with pytest.raises(PotentialError, match="override System atom identities"):
            CABIPotential(library, system=system, config=config)
    with pytest.raises(PotentialError, match="disagree"):
        CABIPotential(library, system={"symbols": ["H"], "atomic_numbers": [8]})


def test_per_frame_failures_remain_visible_and_dynamics_rejects_them(library):
    q = Q.copy()
    q[1] = np.finfo(float).max / 2
    with CABIPotential(library) as potential:
        result = potential.evaluate(q)
        np.testing.assert_array_equal(result.status, [0, 3, 0])
        assert np.isnan(result.energy[1]) and np.isnan(result.forces[1]).all()
        assert np.isnan(result.virial[1]).all()
        assert "nonfinite" in result.messages[1]
        with pytest.raises(PotentialError, match="frames failed"):
            validate_result(result, q.shape)


def test_v2_cannot_export_v1_fused_pointer_and_auto_falls_back(library):
    with CABIPotential(library, system=_system()) as potential:
        with pytest.raises(NativeUnavailableError, match="v2"):
            potential.native_binding()
        initial = State(Q, np.full_like(Q, .03))
        expected = initial.copy()
        dyn = Dynamics(potential, [1837.], 200., len(Q), .1)
        for _ in range(5):
            dyn.step(expected)
        result = dyn.propagate_block(initial, 5)
        assert result.backend == "python"
        np.testing.assert_array_equal(initial.q, expected.q)
        np.testing.assert_array_equal(initial.p, expected.p)


def test_optional_virial_and_nonperiodic_metadata(tmp_path):
    text = SOURCE.read_text().replace(r'\"virial\":true', r'\"virial\":false')
    text = text.replace(r'\"boundary_conditions\":\"periodic\"', r'\"boundary_conditions\":\"nonperiodic\"')
    library = _compile(tmp_path, "no-virial", text)
    with CABIPotential(library) as potential:
        assert potential.evaluate(Q).virial is None
        with pytest.raises(PotentialError, match="nonperiodic"):
            potential.evaluate(Q, cell=CELL)
    with pytest.raises(PotentialError, match="nonperiodic"):
        CABIPotential(library, system=_system())


def test_invalid_metadata_destroys_context_without_v1_fallback(tmp_path):
    text = SOURCE.read_text().replace(r'\"virial_unit\":\"hartree\"', r'\"virial_unit\":\"eV\"')
    text = text.replace("void pyrpmd_pes_destroy_v2(pyrpmd_pes_context_v2 *context) { free(context); }",
                        "static int destroyed = 0;\n"
                        "int test_destroyed(void) { return destroyed; }\n"
                        "void pyrpmd_pes_destroy_v2(pyrpmd_pes_context_v2 *context) { ++destroyed; free(context); }")
    library = _compile(tmp_path, "bad-metadata", text, v1=True)
    raw = ctypes.CDLL(str(library))
    raw.test_destroyed.restype = ctypes.c_int
    with pytest.raises(PotentialError, match="virial_unit"):
        CABIPotential(library)
    assert raw.test_destroyed() == 1
    with CABIPotential(library, abi_version=1) as potential:
        assert potential.abi_version == 1


def test_batch_capability_must_be_boolean(tmp_path):
    text = SOURCE.read_text().replace(r'\"native_batch\":true', r'\"native_batch\":\"yes\"')
    library = _compile(tmp_path, "bad-batch-capability", text)
    with pytest.raises(PotentialError, match="native_batch must be boolean"):
        CABIPotential(library)


@pytest.mark.parametrize("broken", ["status", "force", "virial", "global"])
def test_missing_outputs_and_global_failure_are_detected(tmp_path, broken):
    tail = {
        "status": "y->structure_status[0] = -1;",
        "force": "y->forces[0] = NAN;",
        "virial": "if (y->virials) y->virials[0] = NAN;",
        "global": "message(error, \"backend rejected batch\"); return PYRPMD_BACKEND_FAILURE_V2;",
    }[broken]
    text = SOURCE.read_text()
    # Insert only in evaluate, immediately before its successful return.
    location = text.rfind('    message(error, "");\n    return PYRPMD_OK_V2;')
    text = text[:location] + "    " + tail + "\n" + text[location:]
    library = _compile(tmp_path, "broken-" + broken, text)
    with CABIPotential(library) as potential:
        with pytest.raises(PotentialError):
            potential.evaluate(Q)
