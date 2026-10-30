"""Cross-feature contracts that independent materials/surface branches lacked."""

import json
import numpy as np
import pytest

from pyrpmd import JobSpec, run
from pyrpmd.constants import ANGSTROM_TO_AU
from pyrpmd.cv import make_cv
from pyrpmd.io import fingerprint
from pyrpmd.potentials import PESResult, make_potential
from pyrpmd.potentials.python import PythonPotential, BOHR_ANGSTROM, EV_HARTREE
from pyrpmd.system import System, system_from_config
from pyrpmd.workflows.materials import resolve_materials
from pyrpmd.workflows.site_rate import validate_site_rate


@pytest.mark.parametrize("explicit_flags", [False, True])
def test_saved_nonperiodic_cell_identity_preserves_both_precursor_formats(explicit_flags):
    old = {"symbols": ["H"], "masses": [1837.0], "geometry": [[0.0, 0.0, 0.0]],
           "fragments": [], "charge": 0, "multiplicity": 1,
           "cell": np.diag([4.0, 5.0, 6.0]).tolist()}
    if explicit_flags:
        old["pbc"] = [False, False, False]
    restored = System.from_dict(json.loads(json.dumps(old)))
    assert restored.pbc == (False, False, False)
    assert restored.to_dict() == old
    # Ensure this representation remains stable through multiple checkpoint loads.
    again = System.from_dict(restored.to_dict())
    assert fingerprint({"task": "fes"}, restored) == fingerprint({"task": "fes"}, again)


def test_cell_defaults_are_explicitly_consistent_between_xyz_and_inline(tmp_path):
    xyz = tmp_path / "h.xyz"
    xyz.write_text("1\nHydrogen\nH 0.1 0.2 0.3\n")
    lattice = np.diag([4.0, 5.0, 6.0]).tolist()
    external = system_from_config({"geometry": str(xyz), "cell": lattice})
    inline = system_from_config({"symbols": ["H"], "geometry": [[0.1, 0.2, 0.3]],
                                 "geometry_unit": "angstrom", "cell": lattice})
    np.testing.assert_array_equal(external.cell, inline.cell)
    np.testing.assert_array_equal(external.geometry, inline.geometry)
    np.testing.assert_allclose(external.cell, np.asarray(lattice) * ANGSTROM_TO_AU)
    assert external.pbc == inline.pbc == (True, True, True)


class CallbackPES:
    metadata = {"supports_pbc": True}

    def bind_system(self, system):
        self.cell_angstrom = system.cell * BOHR_ANGSTROM

    def evaluate(self, q):
        phase = 2 * np.pi * (q @ np.linalg.inv(self.cell_angstrom))
        return PESResult(np.cos(phase).sum(axis=(1, 2)),
                         2 * np.pi * np.sin(phase) @ np.linalg.inv(self.cell_angstrom).T,
                         virial=np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy())


class KeywordPES:
    metadata = {"periodic": True}

    def evaluate(self, q, *, cell, pbc):
        assert list(pbc) == [True, True, True]
        phase = 2 * np.pi * (q @ np.linalg.inv(cell))
        return PESResult(np.cos(phase).sum(axis=(1, 2)),
                         2 * np.pi * np.sin(phase) @ np.linalg.inv(cell).T,
                         virial=np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy())


def test_both_explicit_periodic_python_contracts_agree_in_provider_units_and_virial():
    system = System(("D",), [3674.0], [[0.2, 0.3, 0.4]], cell=np.diag([4.0, 5.0, 6.0]))
    units = {"positions": "angstrom", "energy": "eV"}
    callback = make_potential(PythonPotential(CallbackPES(), units=units), system)
    keyword = make_potential(PythonPotential(KeywordPES(), units=units), system)
    q = np.stack([system.geometry, system.geometry + 0.1])
    a, b = callback.evaluate(q), keyword.evaluate(q)
    np.testing.assert_array_equal(a.energy, b.energy)
    np.testing.assert_array_equal(a.forces, b.forces)
    np.testing.assert_array_equal(a.virial, b.virial)
    np.testing.assert_allclose(a.virial, np.broadcast_to(np.eye(3) * EV_HARTREE, (2, 3, 3)))
    shifted = callback.evaluate(q + system.cell[0])
    np.testing.assert_allclose(shifted.energy, a.energy, atol=1e-14)


@pytest.mark.parametrize("task", ["pimd", "diffusion", "site_rate"])
def test_materials_reject_fixed_atoms_before_output_and_in_direct_workflow(task, tmp_path):
    system = System(("H",), [1837.0], [[0.0, 0.0, 0.0]], fixed_atoms=[0])
    mapping = {"task": task, "system": system.to_dict(), "beads": 2,
               "potential": {"provider": "free"},
               "sampling": {"production_steps": 40, "interval_steps": 2}}
    target = tmp_path / task
    with pytest.raises(ValueError, match="all atoms mobile"):
        run(mapping, output=target)
    assert not target.exists()
    guard = validate_site_rate if task == "site_rate" else resolve_materials
    with pytest.raises(ValueError, match="all atoms mobile"):
        guard({"task": task}, system)


def test_free_periodic_provider_and_explicit_geometric_cv_semantics():
    system = System(("H", "H"), [1837, 1837], [[0.1, 0, 0], [3.9, 0, 0]],
                    cell=np.diag([4.0, 5.0, 6.0]), fixed_atoms=[1])
    q = system.geometry[None]
    result = make_potential({"provider": "free"}, system).evaluate(q)
    np.testing.assert_array_equal(result.energy, [0.0])
    assert make_cv({"kind": "distance", "atoms": [0, 1]}, system).value(q)[0] == pytest.approx(3.8)
    assert make_cv({"kind": "periodic_distance", "atoms": [0, 1]}, system).value(q)[0] == pytest.approx(0.2)


@pytest.mark.parametrize("kind", ["distance", "angle", "bond_difference"])
def test_old_implicit_periodic_cv_hash_is_not_accepted_as_new_science(kind):
    """Hash a precursor payload independently: its math was branch-ambiguous."""
    import hashlib
    from pyrpmd.io import jsonable
    from pyrpmd.workflows.fes import _fingerprint

    system = System(("H", "H", "H"), [1837.0] * 3,
                    [[0.1, 0, 0], [3.9, 0.2, 0], [1, 1, 0]], cell=np.eye(3) * 4)
    config = {"task": "fes", "cvs": [{"kind": kind, "atoms": [0, 1] if kind == "distance" else [0, 1, 2]}]}
    legacy_manifest = {"config": config, "system": system.to_dict()}
    legacy_stream = {"schema": 1, "physics": config, "sampling_measure": "cv_density",
                     "system": system.to_dict(), "stream": {"window": 0}}
    def old_hash(payload, *, stream=False):
        options = {"separators": (",", ":"), "default": repr} if stream else {"allow_nan": False}
        return hashlib.sha256(json.dumps(jsonable(payload), sort_keys=True, **options).encode()).hexdigest()
    assert fingerprint(config, system) != old_hash(legacy_manifest)
    assert _fingerprint(config, system, {"window": 0}) != old_hash(legacy_stream, stream=True)
    # Explicit image choices never had that ambiguity and retain their identity.
    config["cvs"][0]["kind"] = "periodic_" + kind
    assert fingerprint(config, system) == old_hash(legacy_manifest)
    assert _fingerprint(config, system, {"window": 0}) == old_hash(legacy_stream, stream=True)
