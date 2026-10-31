"""Real published LEPS model checks; explicitly build its pinned source first."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from benchmarks.brhcl.build_pes import checked_source
from benchmarks.brhcl.pes import BrHClPotential, DEFAULT_LIBRARY, SOURCE, verify
from pyrpmd.bias import UmbrellaBias
from pyrpmd.constants import AMU_TO_AU, ANGSTROM_TO_AU, KCAL_MOL_TO_AU
from pyrpmd.cv import CVStack, Distance
from pyrpmd.dynamics import Dynamics, State, native_capabilities
from pyrpmd.potentials import PotentialError, make_potential
from pyrpmd.propagation import NativeUnavailableError

REAL_PES = pytest.mark.skipif(not DEFAULT_LIBRARY.is_file(), reason="optional pinned BrHCl_LEPS1 has not been built")
NATIVE = pytest.mark.skipif(not native_capabilities()["available"], reason="native PyRPMD build required")
POINT = np.array([[0., 0., 0.], [2.7, .3, 0.], [14., 2., .2]])


def test_source_pin_and_tampered_source_are_checked_before_compilation(tmp_path):
    assert SOURCE["atomic_order"] == ["Br", "H", "Cl"]
    assert SOURCE["atomic_numbers"] == [35, 1, 17]
    assert SOURCE["commit"] == "4716d22e87ae01770cad9ec995c4a02a8e479296"
    assert SOURCE["sha256"] == "55df493416823e16ea26309b474353381e36b982aea95dcc4aaf39097ba02f5e"
    assert SOURCE["morse_dissociation_kcal_mol"] == [90.24, 106.4, 52.09]
    source = tmp_path / "tampered.f"
    source.write_bytes(b" " * SOURCE["bytes"])
    with pytest.raises(ValueError, match="source checksum mismatch"):
        checked_source(source)


@pytest.mark.parametrize("name,value", [("wall_angstrom", -1.), ("wall_angstrom", float("nan")),
                                        ("wall_angstrom", True), ("wall_k_kcal_mol_angstrom2", float("inf")),
                                        ("wall_k_kcal_mol_angstrom2", -2.)])
def test_invalid_wall_configuration_rejected_before_library_load(name, value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        BrHClPotential("missing.so", **{name: value})


@REAL_PES
def test_published_leps_formula_asymptotes_forces_and_rigid_symmetries():
    report = verify()
    assert report["passed"]
    assert report["formula_max_absolute_error_hartree"] < 3e-13
    assert report["force_check"]["max_absolute_force_error"] < 3e-8
    assert report["wall_force_check"]["max_absolute_force_error"] < 2e-9
    expected = [16.16, 0., 54.31]
    actual = [x["energy_hartree"] * SOURCE["legacy_kcal_mol_per_hartree"] for x in report["asymptotic_oracles"]]
    np.testing.assert_allclose(actual, expected, atol=7e-7, rtol=0)


@REAL_PES
def test_mixed_batch_failures_do_not_corrupt_subsequent_valid_frames():
    q = np.stack([POINT, np.zeros((3,3)), [[0.,0.,0.], [1e4,0.,0.], [0.,1e4,0.]], POINT])
    with BrHClPotential() as pes:
        result = pes.evaluate(q)
        np.testing.assert_array_equal(result.status, [0,2,3,0])
        assert np.isnan(result.energy[1:3]).all()
        assert np.isnan(result.forces[1:3]).all()
        np.testing.assert_array_equal(result.energy[[0,3]], np.repeat(pes.evaluate(POINT).energy, 2))
        np.testing.assert_array_equal(result.forces[0], result.forces[3])
        assert "pair distance" in result.messages[1]
        assert "nonfinite" in result.messages[2]


@REAL_PES
def test_native_wall_uses_all_three_pairs_and_preserves_bare_context():
    q = np.array([[0.,0.,0.], [7.,0.,0.], [0.,8.,0.]]) * ANGSTROM_TO_AU
    with BrHClPotential() as bare, BrHClPotential(wall_angstrom=6.) as confined:
        result, physical = confined.evaluate(q), bare.evaluate(q)
        radius_excess = np.array([7., np.sqrt(113), 8.]) - 6.
        expected_wall = .5 * 20. * np.sum(radius_excess**2) * KCAL_MOL_TO_AU
        assert result.energy[0] - physical.energy[0] == pytest.approx(expected_wall, abs=1e-14)
        wall_e, wall_f = confined.wall_energy_force(q)
        np.testing.assert_allclose(result.energy, physical.energy + wall_e, atol=1e-14, rtol=0)
        np.testing.assert_allclose(result.forces, physical.forces + wall_f, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(confined.bare_evaluate(q).energy, physical.energy)
        np.testing.assert_array_equal(confined.bare_evaluate(q).forces, physical.forces)
        # Multiple contexts must not run the in-place source conversion twice.
        np.testing.assert_array_equal(bare.evaluate(q).energy, physical.energy)
        assert confined.metadata["common_pair_wall"]["retained_in_target_pmf"] is True
    with BrHClPotential(wall_angstrom=6., wall_k_kcal_mol_angstrom2=0.) as disabled:
        np.testing.assert_array_equal(disabled.evaluate(q).energy, physical.energy)


@REAL_PES
def test_provenance_binary_mismatch_is_rejected_before_cdll(tmp_path):
    library = tmp_path / "tampered.so"
    shutil.copyfile(DEFAULT_LIBRARY, library)
    report = json.loads(DEFAULT_LIBRARY.with_suffix(".build.json").read_text())
    report["library_sha256"] = "0" * 64
    library.with_suffix(".build.json").write_text(json.dumps(report))
    with pytest.raises(PotentialError, match="provenance checksum mismatch"):
        BrHClPotential(library)


@REAL_PES
@NATIVE
def test_canonical_lock_lifetime_and_native_override_guard(tmp_path):
    alias = tmp_path / "alias.so"
    alias.symlink_to(DEFAULT_LIBRARY)
    original, linked = BrHClPotential(), BrHClPotential(alias, wall_angstrom=6.)
    try:
        assert original._lock is linked._lock
        binding = linked.native_binding()
        assert binding.lock is original._lock
        linked.close()
        with pytest.raises((PotentialError, RuntimeError), match="closed"):
            with binding.acquire():
                pass
        with pytest.raises(PotentialError, match="closed"):
            linked.bare_evaluate(POINT)
        original.evaluate = lambda q: None
        with pytest.raises(NativeUnavailableError, match="overridden"):
            original.native_binding()
    finally:
        original.close()
        linked.close()


@REAL_PES
@NATIVE
def test_actual_native_propagation_with_common_wall_and_two_distance_umbrella():
    rng = np.random.default_rng(645)
    q = np.repeat(POINT[None], 4, axis=0) + rng.normal(scale=.01, size=(4,3,3))
    state = State(q, rng.normal(size=q.shape))
    a, b = state.copy(), state.copy()
    with BrHClPotential(wall_angstrom=6.) as pes:
        dyn = Dynamics(pes, AMU_TO_AU*np.array([79.904, 1.007825, 35.45]), 200., 4, .2, seed=173)
        cv = CVStack([Distance([0,1]), Distance([1,2])])
        bias = UmbrellaBias(cv, [2.7,11.5], [[.02,.003],[.003,.02]])
        native = dyn.propagate_block(a, 25, cv=cv, bias=bias, record_states=True, backend="native")
        python = dyn.propagate_block(b, 25, cv=cv, bias=bias, record_states=True, backend="python")
        assert native.backend == "native"
        np.testing.assert_allclose(native.q, python.q, atol=2e-12, rtol=0)
        np.testing.assert_allclose(native.p, python.p, atol=2e-11, rtol=0)
        np.testing.assert_allclose(native.values, python.values, atol=2e-12, rtol=0)
        np.testing.assert_allclose(native.energies, python.energies, atol=2e-12, rtol=0)
        assert np.ptp(native.energies) < 1e-8


@REAL_PES
@NATIVE
def test_importable_python_factory_forwards_confined_native_context():
    factory = str(Path(__file__).resolve().parents[1] / "benchmarks/brhcl/pes.py") + ":build"
    provider = make_potential({"provider": "python", "factory": factory,
                               "params": {"wall_angstrom": 6.}})
    try:
        with BrHClPotential(wall_angstrom=6.) as direct:
            np.testing.assert_array_equal(provider.evaluate(POINT[None]).energy, direct.evaluate(POINT).energy)
            assert provider.native_binding().context != 0
    finally:
        provider.close()
