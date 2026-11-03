"""Identity and independent force checks for linear-time atom validation."""

import numpy as np
import pytest

from pyrpmd.constants import BOHR_ANGSTROM, EV_TO_AU
from pyrpmd.potentials import PotentialError, check_potential
from pyrpmd.potentials.lammps import LAMMPSPotential


def _model(tmp_path):
    pytest.importorskip("lammps")
    path = tmp_path / "linear-validation.in"
    path.write_text("""units metal
atom_style atomic
boundary p p p
region box block 0 20 0 20 0 20
create_box 3 box
create_atoms 1 single 2 2 2
create_atoms 2 single 3.3 2 2
create_atoms 3 single 5 2 2
mass * 1
pair_style lj/cut 5
pair_coeff * * 0.2 1
neighbor 0.4 bin
""")
    return dict(model=path, symbols=("D", "He", "T"), type_map=("H", "He", "H"),
                cell=np.eye(3)*20/BOHR_ANGSTROM, pbc=(True, True, True))


def _coordinates(count=2):
    q = np.repeat(np.array([[[2., 2, 2], [3.3, 2, 2], [5., 2, 2]]]), count, axis=0)
    q[:, 1, 0] += np.arange(count)*.03
    return q/BOHR_ANGSTROM


def _analytic_lj(q):
    """Small, direct all-pairs reference, independent of LAMMPS atom ordering."""
    energies = np.zeros(len(q))
    forces = np.zeros_like(q)
    for a in range(q.shape[1]):
        for b in range(a):
            delta = (q[:, a]-q[:, b])*BOHR_ANGSTROM
            radius = np.linalg.norm(delta, axis=1)
            s6 = radius**-6
            energies += 4*.2*(s6*s6-s6)*EV_TO_AU
            value = (24*.2/radius**2*(2*s6*s6-s6)*EV_TO_AU*BOHR_ANGSTROM)[:, None]*delta
            forces[:, a] += value
            forces[:, b] -= value
    return energies, forces


@pytest.mark.parametrize("mode", ["serial", "persistent"])
def test_permuted_multispecies_ids_and_type_aliases_preserve_lj_reference(tmp_path, mode):
    options = _model(tmp_path)
    q = _coordinates()
    with LAMMPSPotential(**options, batch_mode=mode) as pes:
        before = pes.evaluate(q)
        # Two types both describe hydrogen (including D/T aliases). Permuting
        # local storage must still validate each type against its global ID.
        for engine in pes.engines:
            for key in ("id", "type", "x", "v", "f", "mask", "image"):
                data = engine.numpy.extract_atom(key)
                data[:3] = data[:3][[2, 0, 1]].copy()
        result = pes.evaluate(q)
        energy, force = _analytic_lj(q)
        np.testing.assert_allclose(result.energy, energy, rtol=2e-13, atol=1e-16)
        np.testing.assert_allclose(result.forces, force, rtol=2e-13, atol=1e-16)
        np.testing.assert_allclose(result.energy, before.energy, rtol=2e-13, atol=1e-16)
        np.testing.assert_allclose(result.forces, before.forces, rtol=2e-13, atol=1e-16)
        assert check_potential(pes, q)["max_absolute_force_error"] < 3e-11


@pytest.mark.parametrize("field,value,match", [
    ("id", 2, "consecutive"),       # Duplicate ID, hence another ID is absent.
    ("id", 0, "consecutive"),
    ("id", 4, "consecutive"),
    ("type", 0, "every atom type"),
    ("type", 4, "every atom type"),
    ("type", 2, "symbols/order"),  # Valid type range but the wrong element.
])
def test_linear_validator_still_rejects_corrupt_ids_and_species(tmp_path, field, value, match):
    with LAMMPSPotential(**_model(tmp_path), batch_mode="persistent") as pes:
        pes.evaluate(_coordinates())
        engine = pes.engines[1]
        ids = engine.numpy.extract_atom("id")[:3]
        row = int(np.flatnonzero(ids == 1)[0])
        engine.numpy.extract_atom(field)[row] = value
        with pytest.raises(PotentialError, match=match):
            pes.evaluate(_coordinates())
        assert pes.closed and pes.engines == ()


def test_cached_box_inverse_updates_with_cell_and_preserves_unwrapped_input(tmp_path):
    options = _model(tmp_path)
    q = _coordinates()
    with LAMMPSPotential(**options, batch_mode="persistent") as pes:
        pes.evaluate(q)
        cell = options["cell"]*1.08
        pes.set_cell(cell)
        q = q*1.08
        reference = pes.evaluate(q)
        q[:, 0] += 3*cell[0]-2*cell[1]+cell[2]
        original = q.copy()
        got = pes.evaluate(q)
        np.testing.assert_array_equal(q, original)
        np.testing.assert_allclose(got.energy, reference.energy, rtol=3e-13, atol=1e-15)
        np.testing.assert_allclose(got.forces, reference.forces, rtol=3e-12, atol=1e-15)
        # Rebinding the same box must update and validate the cached transform.
        pes.bind_system({"symbols": options["symbols"], "cell": cell, "pbc": options["pbc"]})
        np.testing.assert_array_equal(pes.evaluate(q).forces, got.forces)


@pytest.mark.parametrize("mode", ["serial", "persistent"])
def test_vectorized_validator_matches_previous_sort_based_reference(tmp_path, mode):
    # This fixture copies the small old validation method rather than loading
    # Git history, so source distributions exercise the same compatibility gate.
    options = _model(tmp_path)

    class PreviousValidator(LAMMPSPotential):
        def _validate_atoms(self):
            n = len(self.symbols)
            nlocal = self.engine.extract_setting("nlocal")
            assert self.engine.get_natoms() == n == nlocal
            ids = np.asarray(self.engine.numpy.extract_atom("id")[:nlocal], dtype=np.int64)
            assert np.array_equal(np.sort(ids), np.arange(1, n+1))
            types_ = np.asarray(self.engine.numpy.extract_atom("type")[:nlocal], dtype=np.int64)
            assert tuple(self.type_map[int(t)-1] for t in types_[np.argsort(ids)]) == self.symbols
            return ids

    with LAMMPSPotential(**options, batch_mode=mode) as current, PreviousValidator(
            **options, batch_mode=mode) as old:
        assert current.metadata == old.metadata
        assert "neighbor_images" not in current.metadata
        q = _coordinates(1)
        for displacement in (.0, .03, .6):
            q[:, 1, 0] += displacement/BOHR_ANGSTROM
            expected, actual = old.evaluate(q), current.evaluate(q)
            for name in ("energy", "forces", "virial"):
                np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
