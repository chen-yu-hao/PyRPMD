"""Adapter contracts plus optional real-runtime numerical checks."""

import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

from pyrpmd.potentials import PotentialError, check_potential, make_potential
from pyrpmd.potentials.adapters import (
    DeePMDPotential, PySCFPotential, TBLitePotential, make_named_potential,
)
from pyrpmd.potentials.optional import ASEPotential
from pyrpmd.potentials.python import BOHR_ANGSTROM, EV_HARTREE


def _modules(name, **attributes):
    modules = {}
    pieces = name.split(".")
    for count in range(1, len(pieces) + 1):
        key = ".".join(pieces[:count])
        modules[key] = types.ModuleType(key)
    for key, value in attributes.items():
        setattr(modules[name], key, value)
    return modules


class FakeDeepPot:
    instances = []

    def __init__(self, path, **kwargs):
        self.path, self.kwargs = path, kwargs
        self.calls = []
        self.closed = False
        self.instances.append(self)

    def get_type_map(self):
        return ["O", "H"]

    def eval(self, coords, cells, atom_types, **kwargs):
        self.calls.append((coords.copy(), None if cells is None else cells.copy(),
                           atom_types.copy(), kwargs))
        return (np.sum(coords ** 2, axis=(1, 2))[:, None], -2 * coords,
                np.broadcast_to(np.arange(9), (len(coords), 9)).copy())

    def close(self):
        self.closed = True


def test_deepmd_one_native_batch_units_cell_order_and_model_reuse():
    with patch.dict(sys.modules, _modules("deepmd.infer", DeepPot=FakeDeepPot)):
        cell = np.array([[7., .2, .1], [0., 8., .3], [.1, 0., 9.]])
        x = np.arange(18.).reshape(3, 2, 3) / 10
        with DeePMDPotential(["D", "O"], "model.pb", cell=cell) as pes:
            engine = pes.engine
            result = pes.evaluate(x)
            assert len(engine.calls) == 1
            q, c, types_, opts = engine.calls[0]
            np.testing.assert_allclose(q, x * BOHR_ANGSTROM)
            np.testing.assert_allclose(c, np.broadcast_to(cell.ravel() * BOHR_ANGSTROM, (3, 9)))
            np.testing.assert_array_equal(types_, [1, 0])
            assert opts == {"atomic": False}
            np.testing.assert_allclose(result.energy, np.sum(q*q, axis=(1, 2))*EV_HARTREE)
            np.testing.assert_allclose(result.forces, -2*x*BOHR_ANGSTROM**2*EV_HARTREE)
            np.testing.assert_allclose(result.virial, np.broadcast_to(np.arange(9).reshape(3, 3)*EV_HARTREE, (3, 3, 3)))
            pes.evaluate(x[:1])
            assert pes.engine is engine
            assert len(engine.calls) == 2
        assert engine.closed
        with pytest.raises(PotentialError, match="closed"):
            pes.evaluate(x)


def test_deepmd_nonperiodic_and_type_mapping_guards():
    with patch.dict(sys.modules, _modules("deepmd.infer", DeepPot=FakeDeepPot)):
        pes = DeePMDPotential(["H"], "model.pb", cell=np.eye(3), pbc=[False]*3)
        pes.evaluate(np.ones((2, 1, 3)))
        assert pes.engine.calls[0][1] is None
        for mapping in (["H", "H"], ["O"]):
            with pytest.raises(ValueError, match="type_map"):
                DeePMDPotential(["H"], "model.pb", type_map=mapping)
        with pytest.raises(PotentialError, match="partial PBC"):
            DeePMDPotential(["H"], "model.pb", cell=np.eye(3), pbc=[True, False, False])
        with pytest.raises(ValueError, match="output layout"):
            DeePMDPotential(["H"], "model.pb", eval_kwargs={"atomic": True})
        with pytest.raises(ValueError, match="backend"):
            make_named_potential("deepmd", symbols=["H"], model="model.pb", device="cuda:1")


@pytest.mark.parametrize("kind", ["mace", "nequip", "fairchem"])
def test_named_ml_ase_contracts_and_reuse(kind):
    pytest.importorskip("ase")
    from ase.calculators.calculator import Calculator, all_changes

    captured = []

    class CalculatorDouble(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def __init__(self, *args, **kwargs):
            captured.append((args, kwargs))
            super().__init__()
            self.evaluations = []

        @classmethod
        def from_compiled_model(cls, **kwargs):
            return cls(**kwargs)

        @classmethod
        def from_model_checkpoint(cls, *args, **kwargs):
            return cls(*args, **kwargs)

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.evaluations.append(atoms.copy())
            self.results = {"energy": np.sum(atoms.positions**2),
                            "forces": -2*atoms.positions,
                            "stress": np.array([[1., .2, .3], [.2, 2., .4], [.3, .4, 3.]])}

    path, name = {"mace": ("mace.calculators", "MACECalculator"),
                  "nequip": ("nequip.integrations.ase", "NequIPCalculator"),
                  "fairchem": ("fairchem.core", "FAIRChemCalculator")}[kind]
    modules = _modules(path, **{name: CalculatorDouble})
    options = {"task_name": "omol"} if kind == "fairchem" else {}
    cell = np.array([[6., .2, .1], [0., 7., .3], [.1, 0., 8.]])
    with patch.dict(sys.modules, modules):
        system = {"symbols": ["D", "H"], "charge": 1, "multiplicity": 2,
                  "cell": cell, "pbc": [True, False, True]}
        with make_potential({"provider": kind, "model": "custom-model",
                             "device": "cpu", "params": options}, system) as pes:
            x = np.arange(12.).reshape(2, 2, 3) / 10
            calc = pes.atoms.calc
            out = pes.evaluate(x)
            assert len(captured) == 1
            assert len(calc.evaluations) == 2
            assert calc.evaluations[0].get_chemical_symbols() == ["H", "H"]
            assert calc.evaluations[0].info == {"charge": 1, "spin": 2}
            np.testing.assert_array_equal(calc.evaluations[0].pbc, [True, False, True])
            np.testing.assert_allclose(calc.evaluations[0].cell, cell*BOHR_ANGSTROM)
            np.testing.assert_allclose(out.forces, -2*x*BOHR_ANGSTROM**2*EV_HARTREE)
            np.testing.assert_allclose(out.virial[0], -calc.results["stress"]*abs(np.linalg.det(cell*BOHR_ANGSTROM))*EV_HARTREE)
            pes.evaluate(x)
            assert pes.atoms.calc is calc
            assert pes.metadata["native_batch"] is False
            if kind == "mace":
                assert captured[0][1]["model_paths"] == "custom-model"
                assert captured[0][1]["default_dtype"] == "float64"
            elif kind == "nequip":
                assert captured[0][1]["compile_path"] == "custom-model"
                assert captured[0][1]["chemical_species_to_atom_type_map"] is True
            else:
                assert captured[0][0] == ("custom-model",)
                assert captured[0][1]["task_name"] == "omol"
        with pytest.raises(PotentialError, match="closed"):
            pes.evaluate(x)


class FakeTBLite:
    instances = []

    def __init__(self, method, numbers, positions, **kwargs):
        self.method, self.numbers, self.q = method, numbers, positions.copy()
        self.kwargs, self.settings, self.previous = kwargs, {}, []
        self.__class__.instances.append(self)

    def set(self, name, value):
        self.settings[name] = value

    def update(self, positions, lattice=None):
        self.q = positions.copy()
        self.kwargs["lattice"] = lattice

    def singlepoint(self, previous=None):
        self.previous.append(previous)
        return {"energy": np.sum(self.q**2), "gradient": 2*self.q,
                "virial": np.einsum("na,nb->ab", 2*self.q, self.q)}


def test_fairchem_materials_task_uses_zero_spin_embedding_and_rejects_state_selection():
    pytest.importorskip("ase")
    from ase.calculators.lj import LennardJones

    calls = []

    class FakeFAIRChem:
        @classmethod
        def from_model_checkpoint(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return LennardJones()

    modules = _modules("fairchem.core", FAIRChemCalculator=FakeFAIRChem)
    spec = {"provider": "fairchem", "model": "local.pt", "params": {"task_name": "omat"}}
    system = {"symbols": ["Ar"], "cell": np.eye(3)*8, "charge": 0, "multiplicity": 1}
    with patch.dict(sys.modules, modules):
        with make_potential(spec, system) as pes:
            assert pes.atoms.info == {"charge": 0, "spin": 0}
            pes.bind_system(system)
            assert pes.atoms.info == {"charge": 0, "spin": 0}
            assert pes.multiplicity == 1  # System's default, not an actual spin constraint.
        for changed in ({"charge": 1}, {"multiplicity": 2}):
            with pytest.raises(ValueError, match="other than omol"):
                make_potential(spec, dict(system, **changed))
        assert len(calls) == 1  # Invalid states fail before loading a model.


def test_failed_ase_rebind_preserves_provider_and_calculator_cell():
    pytest.importorskip("ase")
    from ase.calculators.lj import LennardJones

    cell = np.eye(3)*8
    with ASEPotential(["Ar"], LennardJones(), cell=cell, pbc=[True, False, True]) as pes:
        with pytest.raises(PotentialError, match="identities differ"):
            pes.bind_system({"symbols": ["Ne"], "cell": cell*2, "pbc": [False]*3})
        np.testing.assert_array_equal(pes.cell, cell)
        np.testing.assert_array_equal(pes.pbc, [True, False, True])
        np.testing.assert_allclose(pes.atoms.cell, cell*BOHR_ANGSTROM)
        np.testing.assert_array_equal(pes.atoms.pbc, [True, False, True])


def test_tblite_native_units_sign_reuse_warm_start_and_rebinding():
    with patch.dict(sys.modules, _modules("tblite.interface", Calculator=FakeTBLite)):
        cell = np.eye(3)*8
        pes = TBLitePotential(["D", "O"], charge=-1, multiplicity=2,
                              cell=cell, pbc=[True, False, True], warm_start=True)
        x = np.arange(12.).reshape(2, 2, 3) / 10
        result = pes.evaluate(x)
        engine = pes.engine
        assert engine.kwargs["charge"] == -1
        assert engine.kwargs["uhf"] == 1
        np.testing.assert_array_equal(engine.numbers, [1, 8])
        np.testing.assert_allclose(result.forces, -2*x)
        np.testing.assert_allclose(result.virial, -2*np.einsum("bna,bnc->bac", x, x))
        assert engine.previous == [None, None]
        pes.evaluate(x*2)
        assert pes.engine is engine
        assert all(v is not None for v in engine.previous[-2:])
        pes.bind_system({"symbols": ["H", "O"], "charge": -1, "multiplicity": 2,
                         "cell": cell*2, "pbc": [True, False, True]})
        pes.evaluate(x)
        assert pes.engine is engine
        np.testing.assert_array_equal(engine.kwargs["lattice"], cell*2)
        with pytest.raises(PotentialError, match="differs"):
            pes.bind_system({"symbols": ["H", "H"]})
        pes.close()
        with pytest.raises(PotentialError, match="closed"):
            pes.evaluate(x)


def test_tblite_failure_resets_warm_start_state():
    class Fail(FakeTBLite):
        def singlepoint(self, previous=None):
            raise RuntimeError("SCC convergence failed")

    with patch.dict(sys.modules, _modules("tblite.interface", Calculator=Fail)):
        pes = TBLitePotential(["H"])
        with pytest.raises(PotentialError, match="frame 0.*SCC"):
            pes.evaluate(np.zeros((1, 1, 3)))
        assert pes.engine is None
        assert pes.results == {}


def test_pyscf_periodic_and_explicit_method_guards():
    with pytest.raises(PotentialError, match="molecules only"):
        PySCFPotential(["H"], cell=np.eye(3))
    pytest.importorskip("pyscf")
    for kwargs, message in (({"method": "DFT"}, "xc"),
                            ({"method": "RHF", "multiplicity": 2}, "singlet"),
                            ({"method": "CCSD"}, "method"),
                            ({"method": "HF", "xc": "pbe"}, "xc")):
        with pytest.raises(ValueError, match=message):
            PySCFPotential(["H"], **kwargs)


def test_real_tblite_h2_force_and_reuse():
    pytest.importorskip("tblite.interface")
    x = np.array([[[0., 0., -.7], [0., 0., .7]], [[0., 0., -.8], [0., 0., .8]]])
    with TBLitePotential(["H", "H"], accuracy=.1) as pes:
        result = pes.evaluate(x)
        engine = pes.engine
        scalar = np.concatenate([pes.evaluate(frame[None]).energy for frame in x])
        assert pes.engine is engine
        np.testing.assert_allclose(result.energy, scalar, atol=1e-10)
        assert check_potential(pes, x[:1], epsilon=1e-4)["max_absolute_force_error"] < 3e-6
        np.testing.assert_allclose(result.forces.sum(axis=1), 0, atol=1e-10)


def test_real_tblite_periodic_cell_strain_virial():
    pytest.importorskip("tblite.interface")
    cell = np.array([[9., .3, .1], [0., 10., .2], [.1, 0., 11.]])
    x = np.array([[[.2, .3, -.7], [.2, .3, .7]]])
    with TBLitePotential(["H", "H"], cell=cell, accuracy=.1) as pes:
        result = pes.evaluate(x)
        eps = 1e-4
        plus = np.eye(3); plus[2, 2] += eps
        minus = np.eye(3); minus[2, 2] -= eps
        pes.bind_system({"symbols": ["H", "H"], "cell": cell @ plus.T})
        ep = pes.evaluate(x @ plus.T).energy
        pes.bind_system({"symbols": ["H", "H"], "cell": cell @ minus.T})
        em = pes.evaluate(x @ minus.T).energy
        np.testing.assert_allclose(result.virial[:, 2, 2], -(ep-em)/(2*eps), atol=3e-6)


def test_real_pyscf_h2_force_reuse_and_scf_failure():
    pyscf = pytest.importorskip("pyscf")
    # Avoid process-wide thread changes; the test harness controls OMP threads.
    x = np.array([[[0., 0., -.7], [0., 0., .7]], [[0., 0., -.8], [0., 0., .8]]])
    with PySCFPotential(["H", "H"], basis="sto-3g", conv_tol=1e-12) as pes:
        out = pes.evaluate(x)
        engine = pes.engine
        scalar = np.concatenate([pes.evaluate(frame[None]).energy for frame in x])
        assert pes.engine is engine
        np.testing.assert_allclose(out.energy, scalar, atol=1e-11)
        assert check_potential(pes, x[:1], epsilon=1e-4)["max_absolute_force_error"] < 2e-7
        np.testing.assert_allclose(out.forces.sum(axis=1), 0, atol=1e-10)
        np.testing.assert_allclose(out.virial, np.einsum("bna,bnc->bac", out.forces, x))
        with patch.object(engine, "kernel", return_value=-1.) as kernel:
            engine.converged = False
            with pytest.raises(PotentialError, match="SCF did not converge"):
                pes.evaluate(x[:1])
        assert pes.engine is None


def test_real_pyscf_open_shell_and_dft():
    pytest.importorskip("pyscf")
    with PySCFPotential(["H"], basis="sto-3g", multiplicity=2) as pes:
        assert pes.method == "UHF"
        out = pes.evaluate(np.zeros((1, 1, 3)))
        assert np.isfinite(out.energy).all()
        np.testing.assert_allclose(out.forces, 0, atol=1e-10)
    with PySCFPotential(["H", "H"], basis="sto-3g", method="DFT", xc="lda,vwn", grid_level=1) as pes:
        assert pes.method == "RKS"
        out = pes.evaluate(np.array([[[0., 0., -.7], [0., 0., .7]]]))
        assert np.isfinite(out.forces).all()


@pytest.mark.parametrize("warm_start", [False, True])
def test_real_pyscf_initial_density_does_not_leak_between_batch_slots(warm_start):
    pytest.importorskip("pyscf")
    x = np.array([[[0., 0., -.7], [0., 0., .7]],
                  [[0., 0., -1.1], [0., 0., 1.1]]])
    with PySCFPotential(["H", "H"], basis="sto-3g", warm_start=warm_start) as pes:
        pes.evaluate(x[:1])
        engine = pes.engine
        assert engine.mo_coeff is not None and engine.mo_occ is not None
        cached = pes.densities[0].copy() if warm_start else None
        original_guess, original_kernel = engine.get_init_guess, engine.kernel
        guesses, supplied = [], []

        def fresh_guess(*args, **kwargs):
            result = original_guess(*args, **kwargs)
            guesses.append(np.asarray(result).copy())
            return result

        def real_kernel(*args, **kwargs):
            # Exercise the actual PySCF driver, including its dm0=None fallback.
            # Every slot must explicitly receive its own initial density.
            assert kwargs.get("dm0") is not None
            supplied.append(np.asarray(kwargs["dm0"]).copy())
            return original_kernel(*args, **kwargs)

        with patch.object(engine, "get_init_guess", side_effect=fresh_guess), \
                patch.object(engine, "kernel", side_effect=real_kernel):
            pes.evaluate(x)

        assert len(supplied) == 2
        if warm_start:
            # Previously sampled slot zero uses its density; newly encountered
            # slot one obtains a fresh guess rather than slot zero's orbitals.
            assert len(guesses) == 1
            np.testing.assert_array_equal(supplied[0], cached)
            np.testing.assert_array_equal(supplied[1], guesses[0])
        else:
            assert len(guesses) == 2
            for actual, fresh in zip(supplied, guesses):
                np.testing.assert_array_equal(actual, fresh)
            assert pes.densities == {}
