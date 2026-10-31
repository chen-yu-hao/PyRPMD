"""Approximate, initially vibration-selected QCT and harmonic NE-RPMD ensembles.

All arrays use the real-mass convention of pyrpmd.dynamics: coordinates are
bohr, momenta are atomic units, and the sum-beads Hamiltonian is sampled at
beta/P. Molecular vibrational modes and ring Fourier modes are separate bases.
The NE-RPMD preparation follows Marjollet et al., JCP 156, 044101 (2022),
doi:10.1063/5.0076216, equations (3)--(7), approximately: a harmonic cold ring
ensemble supplies ZPE, and a centroid displacement adds only nu*omega, not
a second half-quantum. This is not an exact quantum pure-state sampler.

The reference must be an isolated-fragment equilibrium geometry. Both methods
enforce zero *centroid* angular momentum; neither selects arbitrary rotational
quantum states nor assigns final quantum states. Noncentroid free translation
modes are retained for every fragment, including atoms. Rigid rotation ring
modes are omitted. These approximations and the switch to the full PES require
validation for each system. No thermostat or propagation is performed here,
and no global random number generator is used.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dynamics import State
from .units import parse_quantity, unit_factor
from .validation import integer_count as _integer
from .validation import positive_quantity as _positive

_INVERSE_ENERGY = (0, -1, 0, 0, 0, 0)
_SECTION_KEYS = {"method", "beta", "fragments"}
_FRAGMENT_KEYS = {
    "vibrational_quanta", "j", "hessian", "hessian_file", "hessian_unit",
    "frequencies", "frequency_unit", "modes", "modes_file",
}


def _real_array(value, name):
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular real numerical array") from exc
    if result.dtype.kind not in "iuf" or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite real numbers")
    return np.array(result, dtype=float, copy=True, order="C")


def _load_array(path, name):
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError(f"{name} must be a .npy or numerical text file path")
    try:
        if Path(path).suffix.lower() == ".npy":
            result = np.load(path, allow_pickle=False)
        else:
            result = np.loadtxt(path, ndmin=2)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot read {name}: {path}") from exc
    return _real_array(result, name)


def _frequency(value, declared_unit=None):
    # hbar=1: a spectroscopic energy and angular frequency share numerical AU.
    # Wavenumbers mean h*c*nu, not cycles per unit time.
    if declared_unit is not None:
        if isinstance(value, (str, tuple, list)):
            raise ValueError("frequency_unit requires numerical frequencies")
        value = [value, declared_unit]
    try:
        result = parse_quantity(value, "energy")
    except ValueError:
        result = parse_quantity(value, "frequency")
    if result <= 0:
        raise ValueError("Vibrational frequencies must be positive and finite")
    return result


def normalize_initial_ensemble(section):
    """Own and unit-resolve an initial-ensemble mapping without using a PES.

    method is qct or ne_rpmd. The latter requires an explicit beta (numerical
    hartree^-1 or a unit-bearing inverse energy). Two fragment mappings follow
    system.fragments order. A molecule needs either hessian/hessian_file or
    frequencies and modes/modes_file. Explicit mode rows are orthonormal
    mass-weighted molecular eigenvectors [Nv,Nfragment,3] or [Nv,3*Nfragment].

    Numerical frequencies are atomic angular frequencies; frequency_unit may
    specify e.g. cm^-1. Hessians default to hartree/bohr^2. Files are .npy
    (without pickle) or whitespace-delimited numerical text. File contents,
    rather than mutable paths, enter the canonical configuration. Paths must
    already be resolved relative to the input. Omitted quantum numbers mean
    all zero. Atoms use an empty fragment mapping. Hessian-derived modes are
    ordered by increasing frequency; choose an explicit mode basis when a
    particular direction within a degenerate vibrational subspace is intended.
    """
    if not isinstance(section, Mapping):
        raise ValueError("initial_ensemble must be a mapping")
    unknown = set(section) - _SECTION_KEYS
    if unknown:
        raise ValueError(f"Unknown initial_ensemble fields: {sorted(unknown)}")
    method = section.get("method")
    if method not in ("qct", "ne_rpmd"):
        raise ValueError("initial_ensemble.method must be 'qct' or 'ne_rpmd'")
    result = {"method": method}
    if "beta" in section:
        result["beta"] = _positive(section["beta"], "initial_ensemble.beta", _INVERSE_ENERGY)
    elif method == "ne_rpmd":
        raise ValueError("ne_rpmd requires an explicit initial_ensemble.beta")
    specs = section.get("fragments", [{}, {}])
    if not isinstance(specs, (list, tuple)) or len(specs) != 2:
        raise ValueError("initial_ensemble.fragments must contain exactly two mappings")
    fragments = []
    for index, spec in enumerate(specs):
        label = f"initial_ensemble.fragments[{index}]"
        if not isinstance(spec, Mapping):
            raise ValueError(f"{label} must be a mapping")
        unknown = set(spec) - _FRAGMENT_KEYS
        if unknown:
            raise ValueError(f"Unknown {label} fields: {sorted(unknown)}")
        if _integer(spec.get("j", 0), label + ".j") != 0:
            raise ValueError("Only j=0 (zero centroid angular momentum) is supported")
        fragment = {"j": 0}
        if "vibrational_quanta" in spec:
            quantum_numbers = spec["vibrational_quanta"]
            if not isinstance(quantum_numbers, (list, tuple, np.ndarray)):
                raise ValueError(f"{label}.vibrational_quanta must be a list of integers")
            fragment["vibrational_quanta"] = [
                _integer(v, label + ".vibrational_quanta") for v in quantum_numbers
            ]
        hessian_sources = [key for key in ("hessian", "hessian_file") if key in spec]
        mode_sources = [key for key in ("modes", "modes_file") if key in spec]
        if len(hessian_sources) > 1 or len(mode_sources) > 1:
            raise ValueError(f"{label}: choose array or file, not both")
        if hessian_sources and (mode_sources or "frequencies" in spec or "frequency_unit" in spec):
            raise ValueError(f"{label}: choose Hessian or explicit frequencies/modes, not both")
        if "hessian_unit" in spec and not hessian_sources:
            raise ValueError(f"{label}.hessian_unit requires a Hessian")
        if bool(mode_sources) != ("frequencies" in spec):
            raise ValueError(f"{label}: explicit frequencies and modes must be supplied together")
        if "frequency_unit" in spec and "frequencies" not in spec:
            raise ValueError(f"{label}.frequency_unit requires frequencies")
        if hessian_sources:
            key = hessian_sources[0]
            hessian = (_load_array(spec[key], label + ".hessian_file") if key.endswith("_file")
                       else _real_array(spec[key], label + ".hessian"))
            hessian *= unit_factor(str(spec.get("hessian_unit", "hartree/bohr^2")), "spring")
            if not np.isfinite(hessian).all():
                raise ValueError(f"{label}.hessian overflows atomic units")
            fragment.update(hessian=hessian.tolist(), hessian_unit="hartree/bohr^2")
        elif mode_sources:
            values = spec["frequencies"]
            if not isinstance(values, (list, tuple, np.ndarray)):
                raise ValueError(f"{label}.frequencies must be a list")
            frequencies = [_frequency(v, spec.get("frequency_unit")) for v in values]
            key = mode_sources[0]
            modes = (_load_array(spec[key], label + ".modes_file") if key.endswith("_file")
                     else _real_array(spec[key], label + ".modes"))
            fragment.update(frequencies=frequencies, frequency_unit="au", modes=modes.tolist())
        fragments.append(fragment)
    result["fragments"] = fragments
    return result


def _readonly(value):
    array = np.ascontiguousarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


def _rigid_basis(geometry, masses):
    """Mass-orthogonal translations, rotations and their vibrational complement."""
    natoms = len(masses)
    centered = geometry - np.average(geometry, axis=0, weights=masses)
    root_mass = np.sqrt(masses)
    translation = np.zeros((3 * natoms, 3))
    for axis in range(3):
        translation[axis::3, axis] = root_mass / np.sqrt(masses.sum())
    rotation = np.column_stack([
        (np.cross(axis, centered) * root_mass[:, None]).ravel() for axis in np.eye(3)
    ])
    vectors, singular, _ = np.linalg.svd(rotation, full_matrices=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-10)) if singular[0] > 0 else 0
    if natoms > 1 and rank not in (2, 3):
        raise ValueError("Fragment equilibrium geometry must define a linear or nonlinear molecule")
    rigid = np.column_stack((translation, vectors[:, :rank]))
    _, _, right = np.linalg.svd(rigid.T, full_matrices=True)
    vibration = right[3 + rank:].T
    return translation, rigid, vibration, rank


def _ring_basis(beads, beta):
    indices = np.arange(beads)
    rows = [np.ones(beads) / np.sqrt(beads)]
    frequencies = [0.0]
    for k in range(1, (beads + 1) // 2):
        rows.extend((np.sqrt(2.0 / beads) * np.cos(2 * np.pi * k * indices / beads),
                     np.sqrt(2.0 / beads) * np.sin(2 * np.pi * k * indices / beads)))
        frequencies.extend([2 * beads / beta * np.sin(np.pi * k / beads)] * 2)
    if beads % 2 == 0:
        rows.append((-1.0) ** indices / np.sqrt(beads))
        frequencies.append(2 * beads / beta)
    return _readonly(rows), _readonly(frequencies)


@dataclass(frozen=True)
class HarmonicReference:
    """An owned isolated-fragment reference; mode rows use the current masses."""

    atoms: tuple[int, ...]
    masses: np.ndarray
    geometry: np.ndarray
    frequencies: np.ndarray
    modes: np.ndarray
    vibrational_quanta: tuple[int, ...]
    rotational_rank: int
    source: str
    digest: str

    @property
    def nmodes(self):
        return len(self.frequencies)


def _reference(spec, atoms, geometry, masses):
    local_q = np.array(geometry[list(atoms)], copy=True)
    local_mass = np.array(masses[list(atoms)], copy=True)
    _, rigid, complement, rotational_rank = _rigid_basis(local_q, local_mass)
    nv = complement.shape[1]
    if len(atoms) == 1:
        if "hessian" in spec or "modes" in spec:
            raise ValueError("Atomic fragments use free rings and must not specify vibrational references")
        frequencies, modes, source = np.empty(0), np.empty((0, 3)), "free_atom"
    elif "hessian" in spec:
        hessian = _real_array(spec["hessian"], "Cartesian Hessian")
        if hessian.shape != (3 * len(atoms), 3 * len(atoms)):
            raise ValueError("Cartesian Hessian must have shape [3*Nfragment,3*Nfragment]")
        scale = np.max(np.abs(hessian))
        if scale == 0 or not np.allclose(hessian, hessian.T, rtol=0, atol=1e-9 * scale):
            raise ValueError("Cartesian Hessian must be nonzero and symmetric")
        root_mass = np.repeat(np.sqrt(local_mass), 3)
        weighted = 0.5 * (hessian + hessian.T) / np.outer(root_mass, root_mass)
        values, vectors = np.linalg.eigh(complement.T @ weighted @ complement)
        if not np.isfinite(values).all() or np.any(values <= max(np.max(np.abs(values)) * 1e-12, 0.0)):
            raise ValueError("Projected Hessian needs positive vibrational curvature; no imaginary or zero modes")
        frequencies = np.sqrt(values)
        modes = (complement @ vectors).T
        largest = np.argmax(np.abs(modes), axis=1)
        modes *= np.where(modes[np.arange(nv), largest] < 0, -1.0, 1.0)[:, None]
        source = "cartesian_hessian"
    else:
        if "frequencies" not in spec or "modes" not in spec:
            raise ValueError("Each molecular fragment needs a Cartesian Hessian or explicit frequencies and modes")
        frequencies = _real_array(spec["frequencies"], "frequencies")
        modes = _real_array(spec["modes"], "mass-weighted modes")
        if modes.shape == (nv, len(atoms), 3):
            modes = modes.reshape(nv, 3 * len(atoms))
        if frequencies.shape != (nv,) or modes.shape != (nv, 3 * len(atoms)):
            raise ValueError(f"Expected exactly {nv} mass-weighted vibrational modes for this fragment geometry")
        if np.any(frequencies <= 0):
            raise ValueError("Vibrational frequencies must be positive")
        if not np.allclose(modes @ modes.T, np.eye(nv), rtol=0, atol=1e-8):
            raise ValueError("Mass-weighted mode rows must be orthonormal; do not supply Cartesian displacement eigenvectors")
        if np.max(np.abs(modes @ rigid)) > 1e-8:
            raise ValueError("Modes must be orthogonal to mass-weighted translations and rotations for the supplied isotope masses")
        source = "explicit_mass_weighted_modes"
    quanta = tuple(spec.get("vibrational_quanta", [0] * nv))
    if len(quanta) != nv:
        raise ValueError(f"Expected {nv} vibrational_quanta for fragment {atoms}")
    snapshot = {
        "atoms": list(atoms), "geometry_au": local_q.tolist(), "masses_au": local_mass.tolist(),
        "frequencies_au": frequencies.tolist(), "mass_weighted_modes": modes.tolist(),
    }
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return HarmonicReference(atoms, _readonly(local_mass), _readonly(local_q), _readonly(frequencies),
                             _readonly(modes.reshape(nv, len(atoms), 3)), quanta,
                             rotational_rank, source, digest)


def _project_centroid_j0(q, p, masses, target_com):
    """Remove only centroid COM and rigid rotation, preserving internal rings."""
    q += target_com - np.average(q.mean(axis=0), axis=0, weights=masses)
    for _ in range(2):
        mean_p = p.mean(axis=0)
        p -= masses[None, :, None] * (mean_p.sum(axis=0) / masses.sum())
        relative = q.mean(axis=0) - target_com
        mean_p = p.mean(axis=0)
        angular = np.cross(relative, mean_p).sum(axis=0)
        inertia = (np.eye(3) * np.sum(masses * np.sum(relative * relative, axis=1))
                   - np.einsum("a,ai,aj->ij", masses, relative, relative))
        omega = np.linalg.pinv(inertia, rcond=1e-12) @ angular
        p -= masses[None, :, None] * np.cross(omega, relative)[None, :, :]


@dataclass(frozen=True)
class PreparedInitialEnsemble:
    """Reusable reference preparation; independent draws return ordinary State."""

    method: str
    beads: int
    beta: float
    natoms: int
    references: tuple[HarmonicReference, ...]
    ring_transform: np.ndarray
    ring_frequencies: np.ndarray

    @property
    def vibrational_quanta(self):
        return [list(reference.vibrational_quanta) for reference in self.references]

    @property
    def metadata(self):
        fragments = []
        for reference in self.references:
            omega = reference.frequencies
            nu = np.asarray(reference.vibrational_quanta)
            if self.method == "qct":
                expected = (nu + 0.5) * omega
                excess = np.zeros_like(omega)
            else:
                expected = ((omega * omega)[None, :] / (
                    self.beta * (self.ring_frequencies[:, None] ** 2 + omega[None, :] ** 2)
                )).sum(axis=0) + nu * omega
                # Stable without overflowing exp(x).
                x = self.beta * omega
                excess = omega * np.exp(-x) / (-np.expm1(-x))
            fragments.append({
                "atoms": list(reference.atoms), "source": reference.source,
                "reference_sha256": reference.digest, "masses_au": reference.masses.tolist(),
                "equilibrium_geometry_au": reference.geometry.tolist(),
                "mass_weighted_modes": reference.modes.tolist(),
                "mode_order": "increasing_frequency" if reference.source == "cartesian_hessian" else "explicit_input_order",
                "frequencies_au": omega.tolist(), "vibrational_quanta": list(reference.vibrational_quanta),
                "vibrational_mode_count": reference.nmodes, "rotational_rank": reference.rotational_rank,
                "harmonic_target_energy_au": float(np.sum((nu + 0.5) * omega)),
                "expected_mode_energies_before_j0_projection_au": expected.tolist(),
                "infinite_bead_thermal_excess_au": float(excess.sum()),
            })
        return {
            "method": self.method, "beads": self.beads, "beta_au": self.beta,
            "vibrational_quanta": self.vibrational_quanta, "rotation": "j0",
            "free_ring_translation": "all_noncentroid_fragment_com_modes",
            "ring_rotations": "omitted", "mode_convention": "orthonormal_mass_weighted_rows",
            "j0_projection": "rigid_velocity_removed_at_sampled_centroid_geometry_without_rescaling",
            "ring_centroid_convention": "orthogonal_Q0_equals_sqrtP_times_bead_mean",
            "reference_doi": "10.1063/5.0076216",
            "approximation": "harmonic_initial_ensemble; not an exact quantum pure state; no final-state assignment",
            "fragments": fragments,
        }

    def sample(self, rng):
        """Draw one state; caller supplies a trajectory-specific NumPy Generator."""
        if not isinstance(rng, np.random.Generator):
            raise TypeError("sample requires a numpy.random.Generator")
        q = np.empty((self.beads, self.natoms, 3))
        p = np.empty_like(q)
        for reference in self.references:
            omega = reference.frequencies
            nu = np.asarray(reference.vibrational_quanta)
            if self.method == "qct":
                phase = rng.uniform(0, 2 * np.pi, reference.nmodes)
                mode_q = (np.sqrt(2 * (nu + 0.5) / omega) * np.cos(phase))[None, :]
                mode_p = (np.sqrt(2 * (nu + 0.5) * omega) * np.sin(phase))[None, :]
            else:
                variance_p = self.beads / self.beta
                variance_q = variance_p / (omega[None, :] ** 2 + self.ring_frequencies[:, None] ** 2)
                mode_q = rng.normal(size=(self.beads, reference.nmodes)) * np.sqrt(variance_q)
                mode_p = rng.normal(size=mode_q.shape) * np.sqrt(variance_p)
                phase = rng.uniform(0, 2 * np.pi, reference.nmodes)
                mode_q[0] += np.sqrt(2 * self.beads * nu / omega) * np.cos(phase)
                mode_p[0] += np.sqrt(2 * self.beads * nu * omega) * np.sin(phase)
                mode_q = self.ring_transform.T @ mode_q
                mode_p = self.ring_transform.T @ mode_p
            modes = reference.modes.reshape(reference.nmodes, 3 * len(reference.atoms))
            root_mass = np.sqrt(reference.masses)[None, :, None]
            local_q = reference.geometry[None, :, :] + (mode_q @ modes).reshape(self.beads, -1, 3) / root_mass
            local_p = (mode_p @ modes).reshape(self.beads, -1, 3) * root_mass
            if self.method == "ne_rpmd" and self.beads > 1:
                # A free fragment COM has mass M. Keep k>0 ring fluctuations,
                # but fix its centroid; collision launch supplies COM motion.
                translation_q = np.zeros((self.beads, 3))
                translation_p = np.zeros_like(translation_q)
                translation_q[1:] = rng.normal(size=(self.beads - 1, 3)) * np.sqrt(
                    self.beads / (self.beta * self.ring_frequencies[1:, None] ** 2)
                )
                translation_p[1:] = rng.normal(size=(self.beads - 1, 3)) * np.sqrt(self.beads / self.beta)
                total_root = np.sqrt(reference.masses.sum())
                local_q += (self.ring_transform.T @ translation_q)[:, None, :] / total_root
                local_p += ((self.ring_transform.T @ translation_p)[:, None, :]
                            * reference.masses[None, :, None] / total_root)
            target_com = np.average(reference.geometry, axis=0, weights=reference.masses)
            _project_centroid_j0(local_q, local_p, reference.masses, target_com)
            q[:, list(reference.atoms)] = local_q
            p[:, list(reference.atoms)] = local_p
        return State(q, p)

    def diagnostics(self, state):
        """JSON-safe initial/sparse diagnostics, not final quantum-state binning.

        Mass-weighted Eckart/Kabsch alignment removes whole-fragment orientation
        before projecting onto the equilibrium molecular modes. The primitive
        harmonic energy has a *negative* spring term and is distinct from the
        conserved ring Hamiltonian. Individual action estimates may be negative
        or nonintegral; only ensemble/survival comparisons are appropriate.
        """
        if state.q.shape != (self.beads, self.natoms, 3) or state.p.shape != state.q.shape:
            raise ValueError("Diagnostic state shape does not match the prepared ensemble")
        if not np.isfinite(state.q).all() or not np.isfinite(state.p).all():
            raise ValueError("Diagnostic state must be finite")
        fragments = []
        for reference in self.references:
            ids = list(reference.atoms)
            masses = reference.masses
            local_q, local_p = state.q[:, ids], state.p[:, ids]
            mean_q, mean_p = local_q.mean(axis=0), local_p.mean(axis=0)
            center = np.average(mean_q, axis=0, weights=masses)
            momentum = mean_p.sum(axis=0)
            relative = mean_q - center
            ref_center = np.average(reference.geometry, axis=0, weights=masses)
            ref_relative = reference.geometry - ref_center
            if reference.nmodes:
                left, _, right = np.linalg.svd(relative.T @ (masses[:, None] * ref_relative))
                proper = np.eye(3)
                proper[-1, -1] = 1.0 if np.linalg.det(left @ right) >= 0 else -1.0
                rotation = left @ proper @ right
            else:
                rotation = np.eye(3)
            aligned_q = (local_q - center) @ rotation - ref_relative
            aligned_p = (local_p - masses[None, :, None] * momentum / masses.sum()) @ rotation
            modes = reference.modes.reshape(reference.nmodes, 3 * len(ids))
            mq = (aligned_q * np.sqrt(masses)[None, :, None]).reshape(self.beads, -1) @ modes.T
            mp = (aligned_p / np.sqrt(masses)[None, :, None]).reshape(self.beads, -1) @ modes.T
            mq, mp = self.ring_transform @ mq, self.ring_transform @ mp
            omega2 = reference.frequencies[None, :] ** 2
            ring2 = self.ring_frequencies[:, None] ** 2
            physical = np.sum(mp * mp + (omega2 - ring2) * mq * mq, axis=0) / (2 * self.beads)
            centroid_energy = (mp[0] ** 2 + reference.frequencies ** 2 * mq[0] ** 2) / (2 * self.beads)
            gyration = np.sqrt(np.mean(np.sum((local_q - mean_q) ** 2, axis=2), axis=0))
            bead_com = np.sum(local_q * masses[None, :, None], axis=1) / masses.sum()
            fragments.append({
                "atoms": ids, "vibrational_quanta": list(reference.vibrational_quanta),
                "primitive_harmonic_mode_energies_au": physical.tolist(),
                "primitive_harmonic_energy_au": float(physical.sum()),
                "harmonic_mode_actions": (physical / reference.frequencies - 0.5).tolist(),
                "centroid_mode_energies_au": centroid_energy.tolist(),
                "centroid_com_position_au": center.tolist(), "centroid_com_momentum_au": momentum.tolist(),
                "centroid_angular_momentum_au": np.cross(relative, mean_p).sum(axis=0).tolist(),
                "atom_ring_radii_of_gyration_au": gyration.tolist(),
                "fragment_com_ring_spread_au": float(np.sqrt(np.mean(np.sum((bead_com - center) ** 2, axis=1)))),
            })
        return {"method": self.method, "rotation": "j0", "fragments": fragments}


def prepare_initial_ensemble(config_section, system, beads, beta):
    """Validate/diagonalize references once, returning a reusable sampler.

    system may be a System or its dictionary representation. Its masses,
    including isotope overrides, determine Hessian mass weighting and rigid
    projections. Explicit mode data must already correspond to those masses.
    QCT requires one bead; one-bead thermal sampling is not substituted for QCT.
    """
    section = normalize_initial_ensemble(config_section)
    beads = _integer(beads, "beads", 1)
    beta = _positive(beta, "beta", _INVERSE_ENERGY)
    if section["method"] == "qct" and beads != 1:
        raise ValueError("qct initial ensembles require beads=1")
    if section["method"] == "ne_rpmd" and beads < 2:
        raise ValueError("ne_rpmd initial ensembles require beads>=2; use qct for a classical quantum-number ensemble")
    if "beta" in section and not np.isclose(section["beta"], beta, rtol=1e-12, atol=0):
        raise ValueError("initial_ensemble.beta must equal the propagation beta")
    get = system.get if isinstance(system, Mapping) else lambda name, default=None: getattr(system, name, default)
    masses = _real_array(get("masses"), "System masses")
    geometry = _real_array(get("geometry"), "System geometry")
    if masses.ndim != 1 or len(masses) < 2 or np.any(masses <= 0) or geometry.shape != (len(masses), 3):
        raise ValueError("Initial ensemble needs positive masses[N] and geometry[N,3]")
    if any(get("pbc", (False, False, False))) or get("fixed_atoms", ()):
        raise ValueError("State-selected scattering requires a nonperiodic system with all atoms mobile")
    fragments = get("fragments", ())
    if len(fragments) != 2 or any(len(fragment) == 0 for fragment in fragments):
        raise ValueError("State-selected scattering requires exactly two nonempty fragments")
    fragments = tuple(tuple(_integer(atom, "Fragment atom index") for atom in fragment) for fragment in fragments)
    if sorted(atom for fragment in fragments for atom in fragment) != list(range(len(masses))):
        raise ValueError("Fragments must partition all atoms exactly once")
    references = tuple(_reference(spec, fragment, geometry, masses)
                       for spec, fragment in zip(section["fragments"], fragments))
    transform, frequencies = _ring_basis(beads, beta)
    if not np.isfinite(frequencies).all() or not np.isfinite(frequencies ** 2).all():
        raise ValueError("Ring frequencies overflow; check beta and beads")
    for reference in references:
        if (not np.isfinite(reference.frequencies ** 2).all()
                or not np.isfinite(beads / beta / reference.frequencies ** 2).all()):
            raise ValueError("Harmonic variances overflow; check frequencies and beta")
    return PreparedInitialEnsemble(section["method"], beads, beta, len(masses), references, transform, frequencies)
