"""Physical boundary and derivative checks for fixed-cell surface calculations."""

import itertools
import numpy as np
import pytest

from pyrpmd.constants import ANGSTROM_TO_AU
from pyrpmd.cv import PeriodicDistance, SurfaceHeight, SurfaceLateral, make_cv
from pyrpmd.geometry import minimum_image
from pyrpmd.potentials import make_potential, PotentialError, CABIPotential
from pyrpmd.potentials.python import PythonPotential, BOHR_ANGSTROM
from pyrpmd.system import System, system_from_config


def slab(**kwargs):
    values = dict(
        symbols=("H", "Cu", "Cu"), masses=np.array([2.0, 4.0, 6.0]),
        geometry=np.array([[1.0, 0.4, 2.0], [0.0, 0.0, 0.0], [2.0, 1.0, 0.2]]),
        cell=np.array([[4.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 10.0]]),
        pbc=(True, True, False), fixed_atoms=(1, 2),
    )
    values.update(kwargs)
    return System(**values)


def test_system_boundary_units_roundtrip_and_legacy_serialization(tmp_path):
    system = slab()
    restored = System.from_dict(system.to_dict())
    np.testing.assert_array_equal(restored.cell, system.cell)
    assert restored.pbc == system.pbc
    assert restored.fixed_atoms == system.fixed_atoms
    np.testing.assert_array_equal(restored.mobile_mask, [True, False, False])
    restored.cell[0, 0] = 99
    assert system.cell[0, 0] == 4
    inline = system.to_dict()
    inline["cell_unit"] = "angstrom"
    converted = system_from_config(inline)
    np.testing.assert_allclose(converted.cell, system.cell * ANGSTROM_TO_AU)
    np.testing.assert_array_equal(converted.geometry, system.geometry)
    xyz = tmp_path / "slab.xyz"
    xyz.write_text("1\nH on a surface\nH 0 0 2\n")
    file_system = system_from_config({
        "geometry": str(xyz), "cell": system.cell.tolist(), "cell_unit": "bohr", "pbc": [True, True, False],
    })
    # XYZ length is angstrom; explicit cell_unit keeps the numerical cell in bohr.
    np.testing.assert_array_equal(file_system.cell, system.cell)
    assert file_system.geometry[0, 2] == pytest.approx(2 * ANGSTROM_TO_AU)
    nonperiodic = System(("H",), [1], [[0, 0, 0]])
    assert "cell" not in nonperiodic.to_dict()
    assert "pbc" not in nonperiodic.to_dict()
    assert "fixed_atoms" not in nonperiodic.to_dict()


@pytest.mark.parametrize("changes", [
    {"pbc": True}, {"pbc": [1, 1, 0]}, {"pbc": [True, False]},
    {"cell": None}, {"cell": np.zeros((3, 3))}, {"cell": np.eye(2)},
    {"cell": np.eye(3) * np.nan}, {"cell": np.eye(3) * 1j},
    {"fixed_atoms": [True]}, {"fixed_atoms": [0.0]}, {"fixed_atoms": [3]},
    {"fixed_atoms": [-1]}, {"fixed_atoms": [0, 0]},
])
def test_invalid_system_boundaries_fail(changes):
    with pytest.raises(ValueError):
        slab(**changes)


def test_skew_minimum_image_matches_exhaustive_search_and_lattice_translations():
    cell = np.array([[2.0, 0, 0], [1.8, 0.7, 0], [0.3, 0.2, 5.0]])
    rng = np.random.default_rng(328)
    vectors = rng.uniform(-3, 3, size=(37, 3))
    for flags in ((True, True, False), (True, False, True), (True, True, True)):
        periodic = cell[np.asarray(flags)]
        integers = np.array(list(itertools.product(range(-10, 11), repeat=len(periodic))))
        candidates = vectors[:, None] - (integers @ periodic)[None]
        distances = np.linalg.norm(candidates, axis=-1)
        expected = candidates[np.arange(len(vectors)), np.argmin(distances, axis=1)]
        actual = minimum_image(vectors, cell, flags)
        np.testing.assert_allclose(actual, expected, atol=2e-14)
        translated = vectors + np.arange(1, len(periodic) + 1) @ periodic
        np.testing.assert_allclose(minimum_image(translated, cell, flags), actual, atol=2e-14)
    # In-plane translations must never reduce height relative to vacuum.
    np.testing.assert_array_equal(minimum_image(vectors, cell, (True, True, False))[:, 2], vectors[:, 2])


def test_skew_cell_requires_more_than_fractional_rounding():
    cell = np.array([[3.0, 0, 0], [2.7, 0.5, 0], [0, 0, 9.0]])
    vector = np.array([0.49, 0.49, 0.0]) @ cell
    result = minimum_image(vector, cell, (True, True, False))
    assert np.linalg.norm(result) < 0.2 * np.linalg.norm(vector)


def numerical_jacobian(cv, q, epsilon=1e-6):
    result = np.empty_like(q)
    for index in np.ndindex(q.shape):
        plus, minus = q.copy(), q.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        result[index] = (cv.value(plus)[0] - cv.value(minus)[0]) / (2 * epsilon)
    return result


@pytest.mark.parametrize("estimator", ["centroid", "bead_average"])
def test_surface_cvs_gradients_translation_and_ring_scaling(estimator):
    system = slab()
    q = np.repeat(system.geometry[None], 4, axis=0)
    q += np.random.default_rng(4).normal(scale=0.03, size=q.shape)
    for cv in (
        SurfaceHeight([0], system.masses, normal=[0.1, -0.2, 1], reference_atoms=[1, 2], estimator=estimator),
        SurfaceLateral([0], system.masses, direction=[1, 0.3, 0], reference_atoms=[1, 2], estimator=estimator),
        PeriodicDistance([0, 2], system.cell, system.pbc, estimator=estimator),
    ):
        result = cv.value_and_jacobian(q)
        np.testing.assert_allclose(result.jacobian[0], numerical_jacobian(cv, q), atol=5e-10)
        np.testing.assert_allclose(cv.value(q + np.array([8.0, -2.0, 4.0])), result.value, atol=2e-15)
    height = SurfaceHeight([0], system.masses, normal=[0, 0, 2], origin=0.4)
    assert height.value(q)[0] == pytest.approx(q[:, 0, 2].mean() - 0.4)
    np.testing.assert_array_equal(height.centroid_value_gradient_hessian(q.mean(axis=0))[2], 0)
    np.testing.assert_allclose(height.value_and_jacobian(q).jacobian[0, :, 0, 2], 0.25)


def test_surface_cv_factory_defaults_follow_rotated_slab_and_explicit_periodic_distance():
    system = slab(cell=[[0, 4, 0], [0, 1, 3], [10, 0, 0]])
    height = make_cv({"kind": "surface_height", "atoms": [0], "origin": "1 angstrom"}, system)
    np.testing.assert_array_equal(height.normal, [1, 0, 0])
    assert height.origin == pytest.approx(ANGSTROM_TO_AU)
    lateral = make_cv({"kind": "surface_lateral", "atoms": [0]}, system)
    np.testing.assert_array_equal(lateral.direction, [0, 1, 0])
    q = np.array([[[0, 0, 0], [0, 3.8, 0], [0, 0, 1]]], float)
    plain = make_cv({"kind": "distance", "atoms": [0, 1]}, system)
    periodic = make_cv({"kind": "periodic_distance", "atoms": [0, 1]}, system)
    assert plain.value(q)[0] == pytest.approx(3.8)
    assert periodic.value(q)[0] == pytest.approx(0.2)


def test_custom_cv_receives_readonly_boundary_metadata(tmp_path):
    plugin = tmp_path / "surface_cv.py"
    plugin.write_text(
        "import numpy as np\n"
        "class CV:\n"
        "    metadata = {'dimension': 1, 'units': ('bohr',), 'periods': (None,)}\n"
        "    def value_and_jacobian(self, state):\n"
        "        assert state.pbc == (True, True, False)\n"
        "        assert state.fixed_atoms == (1, 2)\n"
        "        assert not state.cell.flags.writeable\n"
        "        assert not state.q.flags.writeable\n"
        "        assert state.cell[0, 0] == 4.0\n"
        "        jac = np.zeros((1,)+state.q.shape)\n"
        "        jac[0, :, 0, 2] = 1/len(state.q)\n"
        "        return np.array([state.q[:,0,2].mean()]), jac\n"
        "def factory():\n"
        "    return CV()\n"
    )
    system = slab()
    cv = make_cv({"kind": "python", "factory": str(plugin) + ":factory"}, system)
    q = np.repeat(system.geometry[None], 4, axis=0)
    assert cv.value(q)[0] == pytest.approx(2.0)


@pytest.mark.parametrize("spec", [
    {"atoms": []}, {"atoms": [True]}, {"atoms": [0, 0]}, {"atoms": [3]},
    {"atoms": [0], "reference_atoms": [0]}, {"atoms": [0], "reference_atoms": []},
    {"atoms": [0], "normal": [0, 0, 0]}, {"atoms": [0], "normal": [1, 2]},
    {"atoms": [0], "origin": np.nan},
])
def test_surface_cv_invalid_groups_and_frames(spec):
    with pytest.raises(ValueError):
        SurfaceHeight(masses=slab().masses, **spec)


class PeriodicToy:
    metadata = {"supports_pbc": True}

    def bind_system(self, system):
        self.system = system

    def evaluate(self, q):
        return np.zeros(len(q)), np.zeros_like(q)


def test_periodic_pes_requires_explicit_contract_and_receives_system():
    system = slab()
    # The direct public constructor must reject PBC before loading a library.
    for source in (system, system.to_dict()):
        with pytest.raises(PotentialError, match="nonperiodic"):
            CABIPotential("does-not-exist.so", system=source)
    provider = make_potential(PeriodicToy(), system)
    assert provider.system is system
    wrapped = make_potential(PythonPotential(PeriodicToy()), system)
    assert wrapped.obj.system is system
    from pyrpmd.propagation import NativeUnavailableError

    with pytest.raises(NativeUnavailableError, match="periodic"):
        wrapped.native_binding()
    for provider in (
        {"provider": "cabi", "library": "does-not-exist.so"},
        {"provider": "torch", "model": "does-not-exist.pt"},
        lambda q: (np.zeros(len(q)), np.zeros_like(q)),
    ):
        with pytest.raises(PotentialError, match="periodic|Periodic|PBC"):
            make_potential(provider, system)
    missing_callback = lambda q: (np.zeros(len(q)), np.zeros_like(q))
    missing_callback.metadata = {"supports_pbc": True}
    with pytest.raises(PotentialError, match="bind_system"):
        make_potential(missing_callback, system)


def test_periodic_python_binding_survives_process_bead_distribution(tmp_path):
    from pyrpmd.parallel import ProcessBeadPotential

    plugin = tmp_path / "periodic_pes.py"
    plugin.write_text(
        "import numpy as np\n"
        "class PeriodicPES:\n"
        "    metadata = {'supports_pbc': True}\n"
        "    def bind_system(self, system):\n"
        "        assert system.pbc == (True, True, False)\n"
        "        assert system.fixed_atoms == (1, 2)\n"
        "        self.inverse = np.linalg.inv(system.cell)\n"
        "    def evaluate(self, q):\n"
        "        phase = 2*np.pi*(q @ self.inverse)[..., :2]\n"
        "        energy = np.cos(phase).sum(axis=(1,2))\n"
        "        force = 2*np.pi*np.sin(phase) @ self.inverse[:, :2].T\n"
        "        return energy, force\n"
        "def factory():\n"
        "    return PeriodicPES()\n"
    )
    system = slab()
    spec = {"provider": "python", "factory": str(plugin) + ":factory"}
    q = np.repeat(system.geometry[None], 5, axis=0)
    q += np.random.default_rng(9).normal(scale=0.2, size=q.shape)
    serial = make_potential(spec, system)
    distributed = ProcessBeadPotential(spec, system, 2)
    try:
        reference = serial.evaluate(q)
        actual = distributed.evaluate(q)
        np.testing.assert_array_equal(actual.energy, reference.energy)
        np.testing.assert_array_equal(actual.forces, reference.forces)
        shifted = distributed.evaluate(q + system.cell[1])
        np.testing.assert_allclose(shifted.energy, reference.energy, atol=2e-14)
        np.testing.assert_allclose(shifted.forces, reference.forces, atol=2e-14)
    finally:
        distributed.close()
        serial.close()


def test_ase_cell_pbc_units_and_force_evaluation():
    pytest.importorskip("ase")
    from ase.calculators.calculator import Calculator, all_changes
    from pyrpmd.potentials.optional import ASEPotential

    class InspectCell(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            assert atoms.pbc.tolist() == [True, True, False]
            np.testing.assert_allclose(atoms.cell.array, system.cell * BOHR_ANGSTROM)
            self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}

    system = slab()
    potential = make_potential(ASEPotential(system.symbols, InspectCell()), system)
    result = potential.evaluate(system.geometry[None])
    np.testing.assert_array_equal(result.forces, np.zeros((1, 3, 3)))
    np.testing.assert_allclose(potential.atoms.positions, system.geometry * BOHR_ANGSTROM)
