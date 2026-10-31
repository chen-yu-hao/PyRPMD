"""Published BrHCl_LEPS1 PES, analytic forces and optional common confinement.

Coordinates are bohr in fixed [Br,H,Cl] order. Results are batches, even for
one [3,3] frame. The default is the bare PES. The all-pair wall is a physical
addition retained in the target distribution; it is never an umbrella bias.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pyrpmd.constants import ANGSTROM_TO_AU, KCAL_MOL_TO_AU
from pyrpmd.potentials import Potential, PotentialError, CABIPotential, check_potential
from pyrpmd.potentials.base import coordinates

HERE = Path(__file__).resolve().parent
SOURCE = json.loads((HERE / "source.json").read_text())
DEFAULT_LIBRARY = HERE / "build" / "libbrhcl.so"
_DOUBLE_ARRAY = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_INT_ARRAY = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")


def _coordinates(q):
    q = np.asarray(q)
    if q.ndim == 2:
        q = q[None]
    q = coordinates(q)
    if q.shape[1:] != (3, 3):
        raise PotentialError("BrHCl requires three atoms in [Br,H,Cl] order")
    return q


def _nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


class BrHClPotential(Potential):
    def __init__(self, library=None, *, wall_angstrom=0.0, wall_k_kcal_mol_angstrom2=20.0):
        self.wall_angstrom = _nonnegative(wall_angstrom, "wall_angstrom")
        self.wall_k_kcal_mol_angstrom2 = _nonnegative(wall_k_kcal_mol_angstrom2, "wall_k_kcal_mol_angstrom2")
        self.path = Path(library or DEFAULT_LIBRARY).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Build the pinned BrHCl PES first: python {HERE / 'build_pes.py'}")
        provenance = self.path.with_suffix(".build.json")
        if not provenance.is_file():
            raise PotentialError(f"Missing BrHCl build provenance: {provenance}")
        self.provenance = json.loads(provenance.read_text())
        binary_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if (self.provenance.get("source", {}).get("sha256") != SOURCE["sha256"]
                or self.provenance.get("library_sha256") != binary_hash):
            raise PotentialError("BrHCl library/source provenance checksum mismatch")
        for name in ("wrapper.f90", "wrapper.cpp", "exports.map"):
            current = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            if self.provenance.get("wrapper_sha256", {}).get(name) != current:
                raise PotentialError("BrHCl wrapper changed since compilation; rebuild the library")
        self._radius = self.wall_angstrom * ANGSTROM_TO_AU
        self._kappa = self.wall_k_kcal_mol_angstrom2 * KCAL_MOL_TO_AU / ANGSTROM_TO_AU**2
        self._provider = CABIPotential(self.path, config={
            "wall_radius_bohr": self._radius, "wall_k_hartree_bohr2": self._kappa,
            "atomic_numbers": [35, 1, 17], "charge": 0, "multiplicity": 2,
        })
        self._lock = self._provider._lock
        self._bare_provider = None
        self.closed = False
        self.metadata = dict(self._provider.metadata)
        self.metadata.update({"energy_zero": SOURCE["energy_zero"], "source_commit": SOURCE["commit"],
                              "source_sha256": SOURCE["sha256"], "library_sha256": binary_hash,
                              "minimum_pair_distance_bohr": SOURCE["minimum_pair_distance_bohr"],
                              "wall_angstrom": self.wall_angstrom,
                              "wall_k_kcal_mol_angstrom2": self.wall_k_kcal_mol_angstrom2})
        self._radial = self._provider.lib.pyrpmd_brhcl_radial
        self._radial.argtypes = [_DOUBLE_ARRAY, _DOUBLE_ARRAY, _DOUBLE_ARRAY, _INT_ARRAY]
        self._radial.restype = None

    def evaluate(self, q):
        return self._provider.evaluate(_coordinates(q))

    def native_binding(self):
        from pyrpmd.propagation import NativeUnavailableError
        if type(self) is not BrHClPotential or "evaluate" in self.__dict__:
            raise NativeUnavailableError("overridden BrHCl provider uses Python semantics")
        return self._provider.native_binding()

    def bare_evaluate(self, q):
        """Evaluate the source PES without confinement, with identical atom order."""
        q = _coordinates(q)
        with self._lock:
            if self.closed:
                raise PotentialError("PES context is closed")
            if self._radius == 0 or self._kappa == 0:
                return self._provider.evaluate(q)
            if self._bare_provider is None:
                self._bare_provider = CABIPotential(self.path)
            return self._bare_provider.evaluate(q)

    def wall_energy_force(self, q):
        """Independent NumPy oracle for the common wall, in atomic units."""
        q = _coordinates(q)
        energy, force = np.zeros(len(q)), np.zeros_like(q)
        if self._radius == 0 or self._kappa == 0:
            return energy, force
        for a, b in ((0, 1), (1, 2), (0, 2)):
            delta = q[:, a] - q[:, b]
            distance = np.linalg.norm(delta, axis=1)
            excess = np.maximum(distance - self._radius, 0)
            energy += .5 * self._kappa * excess**2
            factor = np.divide(-self._kappa * excess, distance, out=np.zeros_like(distance), where=distance > 0)
            contribution = factor[:, None] * delta
            force[:, a] += contribution
            force[:, b] -= contribution
        return energy, force

    def radial(self, distances):
        """Bare E and +dE/dr; distances order (Br-H,H-Cl,Br-Cl), all in bohr."""
        r = np.ascontiguousarray(distances, dtype=np.float64)
        if r.shape != (3,):
            raise ValueError("radial input must contain three distances")
        energy, gradient, status = np.empty(1), np.empty(3), np.zeros(1, dtype=np.int32)
        with self._lock:
            if self.closed:
                raise PotentialError("PES context is closed")
            self._radial(r, energy, gradient, status)
        if status[0]:
            raise PotentialError(f"Invalid BrHCl radial geometry, status={status[0]}")
        return float(energy[0]), gradient

    def close(self):
        if not hasattr(self, "_provider"):
            return
        with self._lock:
            self._provider.close()
            if self._bare_provider is not None:
                self._bare_provider.close()
            self.closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def build(library=None, **kwargs):
    """Importable factory for PyRPMD Python-provider configuration."""
    return BrHClPotential(library, **kwargs)


def analytic_leps_energy(distances):
    """Independent NumPy energy formula from the pinned source parameters.

    This validation oracle retains the source's rounded internal sqrt(2).
    It is not used to run dynamics or reconstruct the independent quadrature.
    """
    distances = np.asarray(distances, dtype=float)
    if distances.shape[-1:] != (3,):
        raise ValueError("final axis must be (Br-H,H-Cl,Br-Cl)")
    d = np.array(SOURCE["morse_dissociation_kcal_mol"]) / SOURCE["legacy_kcal_mol_per_hartree"]
    re = np.array(SOURCE["morse_equilibrium_angstrom"]) / SOURCE["legacy_angstrom_per_bohr"]
    beta = np.array(SOURCE["morse_beta_inverse_angstrom"]) * SOURCE["legacy_angstrom_per_bohr"]
    z = np.array(SOURCE["sato_parameters"])
    x = np.exp(-beta * (distances - re))
    coulomb = d / (4*(1+z)) * ((z+3)*x - 2*(1+3*z)) * x
    exchange = d / (4*(1+z)) * ((1+3*z)*x - 2*(z+3)) * x
    radical = np.sqrt(np.sum((exchange - np.roll(exchange, 1, axis=-1))**2, axis=-1))
    return np.sum(coulomb, axis=-1) - radical/SOURCE["legacy_sqrt_two"] + d[1]


def verify(library=None):
    """Source formula, asymptotes, Cartesian/radial forces and native wall."""
    with BrHClPotential(library) as bare, BrHClPotential(library, wall_angstrom=6.) as confined:
        rng = np.random.default_rng(65029)
        points = rng.normal(size=(40, 3, 3)) * 1.4
        points[:, 0, 0] -= 3.5
        points[:, 2, 0] += 3.5
        distances = np.stack([np.linalg.norm(points[:, a] - points[:, b], axis=-1) for a, b in ((0,1),(1,2),(0,2))], axis=1)
        result = bare.evaluate(points)
        if np.any(result.status):
            raise AssertionError("Oracle geometry unexpectedly outside source domain")
        formula_error = float(np.max(abs(result.energy - analytic_leps_energy(distances))))
        if formula_error > 3e-13:
            raise AssertionError(f"Independent LEPS energy mismatch: {formula_error}")
        force_check = check_potential(bare, points, epsilon=1e-5)
        if force_check["max_absolute_force_error"] > 3e-8:
            raise AssertionError(f"Cartesian force mismatch: {force_check}")
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        moved = bare.evaluate(points @ rotation + [7., -3., 1.])
        rotation_energy_error = float(np.max(abs(moved.energy - result.energy)))
        rotation_force_error = float(np.max(abs(moved.forces - result.forces @ rotation)))
        np.testing.assert_allclose(moved.energy, result.energy, atol=3e-13, rtol=0)
        np.testing.assert_allclose(moved.forces, result.forces @ rotation, atol=3e-12, rtol=0)
        net_force = float(np.max(abs(np.sum(result.forces, axis=1))))
        net_torque = float(np.max(abs(np.sum(np.cross(points, result.forces), axis=1))))
        if net_force > 1e-12 or net_torque > 1e-11:
            raise AssertionError("Net force/torque violates rigid-motion invariance")
        scalar = [bare.evaluate(q) for q in points]
        np.testing.assert_array_equal(result.energy, np.concatenate([r.energy for r in scalar]))
        np.testing.assert_array_equal(result.forces, np.concatenate([r.forces for r in scalar]))
        r = np.array([3.3, 2.9, 4.7])
        energy, gradient = bare.radial(r)
        eps = 1e-5
        numerical = np.array([(bare.radial(r+eps*np.eye(3)[i])[0] - bare.radial(r-eps*np.eye(3)[i])[0])/(2*eps) for i in range(3)])
        radial_error = float(np.max(abs(gradient - numerical)))
        if radial_error > 2e-9:
            raise AssertionError(f"Radial force mismatch: {radial_error}")
        extended = np.array([[[0.,0.,0.], [2.7,.3,0.], [14.,2.,.2]],
                             [[0.,0.,0.], [12.,.1,0.], [2.,3.7,.2]]])
        pe, we = bare.evaluate(extended), confined.evaluate(extended)
        wall_e, wall_f = confined.wall_energy_force(extended)
        np.testing.assert_allclose(we.energy, pe.energy+wall_e, atol=2e-15, rtol=0)
        np.testing.assert_allclose(we.forces, pe.forces+wall_f, atol=2e-15, rtol=0)
        wall_force_check = check_potential(confined, extended, epsilon=1e-5)
        if wall_force_check["max_absolute_force_error"] > 2e-9:
            raise AssertionError(f"Confined force mismatch: {wall_force_check}")
        independent_bare = confined.bare_evaluate(extended)
        np.testing.assert_array_equal(independent_bare.energy, pe.energy)
        batch = np.repeat(points, 100, axis=0)
        started = time.perf_counter()
        bare.evaluate(batch)
        elapsed = time.perf_counter() - started
        return {
            "schema": "pyrpmd.brhcl.leps1-verification.v1", "passed": True,
            "source": SOURCE, "build": bare.provenance,
            "asymptotic_oracles": bare.provenance["asymptotic_oracles"],
            "formula_points": len(points), "formula_max_absolute_error_hartree": formula_error,
            "force_check": force_check, "radial_max_absolute_derivative_error": radial_error,
            "rotation_translation_energy_error_hartree": rotation_energy_error,
            "rotation_translation_force_error_hartree_per_bohr": rotation_force_error,
            "net_force_max_hartree_per_bohr": net_force, "net_torque_max_hartree": net_torque,
            "batch_scalar_exact": True, "confined_bare_isolation": True,
            "common_pair_wall": confined.metadata["common_pair_wall"], "wall_force_check": wall_force_check,
            "benchmark_frames": len(batch), "microseconds_per_frame": elapsed*1e6/len(batch),
            "scope": "Pinned historical LEPS model and adapter verification; not a claim of quantitative H+BrCl branching accuracy.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.library)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
