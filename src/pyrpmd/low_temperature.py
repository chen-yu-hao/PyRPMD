"""Hessian-based cross-checks and multidimensional thermal instanton rates.

Atomic units throughout (bohr, electron masses, hartree, hbar=1), except
explicitly named temperatures and SI outputs. See docs/low-temperature.md
for the determinant, partition-function and standard-state conventions.

References: Richardson & Althorpe, JCP 131, 214106 (2009),
10.1063/1.3267318; Richardson, JCP 144, 114106 (2016),
10.1063/1.4943866; Eckart, Phys. Rev. 35, 1303 (1930),
10.1103/PhysRev.35.1303.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import sys

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from .constants import ATOMIC_TIME_S, BOHR_M, KB_AU
from .instanton import (InstantonError, RingPolymerAction, _mask_and_masses,
                       _physical_hessians, _positive)
from .potentials.base import coordinates, evaluate_checked

_LOG2 = math.log(2.0)


def _matrix(value, size, name):
    a = np.asarray(value)
    if (a.dtype.kind not in "fiu" or a.shape != (size, size)
            or not np.isfinite(a).all()):
        raise ValueError(f"{name} must be a finite real ({size},{size}) matrix")
    if not np.allclose(a, a.T, rtol=1e-8, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    a = np.array(a, dtype=float, copy=True)
    return (a + a.T) * 0.5


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _complement(basis):
    """Orthonormal complement of an already independent column basis."""
    if basis.shape[1] == 0:
        return np.eye(basis.shape[0])
    return np.linalg.qr(basis, mode="complete")[0][:, basis.shape[1]:]


def _rigid_basis(path, masses):
    """Global, not beadwise, mass-weighted translations and rotations."""
    q = np.asarray(path).reshape(-1, 3)
    m = np.tile(masses, len(path))
    center = np.sum(m[:, None] * q, axis=0) / np.sum(m)
    x = q - center
    translations = np.array([(np.sqrt(m)[:, None] * axis).ravel()
                             for axis in np.eye(3)]).T
    translations /= np.linalg.norm(translations, axis=0)
    rotations = np.array([(np.sqrt(m)[:, None] * np.cross(axis, x)).ravel()
                          for axis in np.eye(3)]).T
    u, singular, _ = np.linalg.svd(rotations, full_matrices=False)
    rank = int(np.count_nonzero(singular**2 > max(singular[0]**2 * 1e-8, 1e-14)))
    return np.column_stack((translations, u[:, :rank]))


def _external_logs(path, masses, beta, symmetry_number):
    """Translation per bohr^3 and classical rigid rotation of a supermolecule.

    For a P-bead ring, pass the ENTIRE path and beta/P. Both total mass and
    inertia contain all P beads. Nonzero ring modes cancel the extra P powers.
    """
    q = np.asarray(path).reshape(-1, 3)
    m = np.tile(masses, len(path))
    center = np.sum(m[:, None] * q, axis=0) / np.sum(m)
    x = q - center
    inertia = np.eye(3) * np.sum(m[:, None] * x*x) - (x.T * m) @ x
    moments = np.linalg.eigvalsh(inertia)
    cutoff = max(float(moments[-1]) * 1e-8, 1e-14)
    positive = moments[moments > cutoff]
    if len(positive) == 0:
        if symmetry_number != 1:
            raise ValueError("an atom has rotational symmetry_number=1")
        rotation = 0.0
    elif len(positive) == 2:
        rotation = math.log(2 / beta) + float(np.log(positive).mean()) - math.log(symmetry_number)
    elif len(positive) == 3:
        rotation = (0.5 * math.log(math.pi) + 1.5 * math.log(2 / beta)
                    + 0.5 * float(np.log(positive).sum()) - math.log(symmetry_number))
    else:
        raise InstantonError("invalid inertia tensor")
    translation = 1.5 * math.log(float(m.sum()) / (2 * math.pi * beta))
    return translation, rotation


def _check_null(matrix, basis, tolerance=1e-5, scale=None):
    scale = np.linalg.norm(matrix) if scale is None else scale
    if basis.shape[1] and np.max(np.linalg.norm(matrix @ basis, axis=0)) > tolerance * max(scale, 1e-30):
        raise InstantonError("Hessian does not have the requested rigid-body zero modes")


@dataclass
class HarmonicState:
    """Stationary-point data, including an UNWEIGHTED active Cartesian Hessian.

    ``rigid_body=True`` means an isolated gas molecule with every Cartesian
    coordinate active. Otherwise this is a bound/constrained Cartesian model;
    no translation or rotation is silently removed. Symmetry and electronic
    degeneracy must be supplied, not guessed from coordinates or atom labels.
    """
    geometry_bohr: np.ndarray
    masses: np.ndarray
    energy_hartree: float
    hessian: np.ndarray
    active_mask: np.ndarray | None = None
    rigid_body: bool = False
    symmetry_number: int = 1
    electronic_degeneracy: int = 1

    def __post_init__(self):
        self.geometry_bohr = coordinates(np.asarray(self.geometry_bohr)[None])[0].copy()
        self.active_mask, mass = _mask_and_masses(self.geometry_bohr.shape, self.masses, self.active_mask)
        self.masses = np.array(self.masses, dtype=float, copy=True)
        self.hessian = _matrix(self.hessian, len(mass), "hessian")
        self.energy_hartree = float(self.energy_hartree)
        if not math.isfinite(self.energy_hartree):
            raise ValueError("energy_hartree must be finite")
        if not isinstance(self.rigid_body, (bool, np.bool_)):
            raise ValueError("rigid_body must be boolean")
        self.symmetry_number = _integer(self.symmetry_number, "symmetry_number")
        self.electronic_degeneracy = _integer(self.electronic_degeneracy, "electronic_degeneracy")
        if self.rigid_body and not self.active_mask.all():
            raise ValueError("rigid_body requires all Cartesian coordinates active")
        if not self.rigid_body and self.symmetry_number != 1:
            raise ValueError("constrained states do not have a rotational symmetry number")

    @classmethod
    def from_potential(cls, potential, geometry_bohr, masses, *, hessian=None,
                       active_mask=None, hessian_step_bohr=1e-4,
                       force_tolerance=1e-6, **options):
        """Reuse a supplied Hessian, the PES provider, or finite-difference forces.

        A callable ``potential.hessian(x)`` supplies unweighted Cartesian PES
        Hessians in atomic units; its validation errors are never hidden by a
        finite-difference fallback. See :mod:`pyrpmd.instanton` for the contract.
        """
        q = coordinates(np.asarray(geometry_bohr)[None])
        mask, _ = _mask_and_masses(q.shape[1:], masses, active_mask)
        value = evaluate_checked(potential, q)
        energy = float(value.energy[0])
        if np.max(np.abs(value.forces[:, mask])) > _positive(force_tolerance, "force_tolerance"):
            raise InstantonError("harmonic state is not stationary in active coordinates")
        if hessian is None:
            hessian = _physical_hessians(potential, q, mask, _positive(hessian_step_bohr, "hessian_step_bohr"))[0]
        return cls(q[0], masses, energy, hessian, mask, **options)

    def spectrum(self, negative_modes=0):
        """Return positive omega^2, |imaginary omega| and removed rigid count."""
        _, mass = _mask_and_masses(self.geometry_bohr.shape, self.masses, self.active_mask)
        h = self.hessian / np.sqrt(mass[:, None] * mass[None, :])
        basis = (_rigid_basis(self.geometry_bohr[None], self.masses) if self.rigid_body
                 else np.empty((len(mass), 0)))
        _check_null(h, basis)
        complement = _complement(basis)
        eig = np.linalg.eigvalsh(complement.T @ h @ complement)
        cutoff = max(float(np.max(np.abs(eig), initial=0)) * 1e-8, 1e-30)
        if np.any(np.abs(eig) <= cutoff):
            raise InstantonError("unresolved internal zero mode; a harmonic partition is undefined")
        negative = eig[eig < -cutoff]
        if len(negative) != negative_modes:
            raise InstantonError(f"expected {negative_modes} negative modes, found {len(negative)}")
        omega = math.sqrt(-float(negative[0])) if len(negative) else 0.0
        return eig[eig > cutoff], omega, basis.shape[1]

    def log_partition(self, beta, *, classical=False, nbeads=None, saddle=False):
        """Partition without exp(-beta*electronic_energy); gas value per bohr^3.

        A finite nbeads uses the same primitive discretization as the instanton.
        No finite-P saddle partition or classical finite-P combination is used.
        """
        return self.partition_diagnostics(beta, classical=classical,
                                          nbeads=nbeads, saddle=saddle)["log_partition"]

    def partition_diagnostics(self, beta, *, classical=False, nbeads=None, saddle=False):
        """Return the partition and its vibrational, rigid and electronic parts.

        The electronic *energy* Boltzmann factor is excluded. For finite P,
        the exact primitive harmonic determinant is evaluated in closed form,
        with work independent of P. Global rigid modes are removed geometrically;
        the noncentroid free-ring factors cancel the supermolecule bead powers.
        ``log_translation_partition`` is per bohr cubed for an isolated gas.
        """
        beta = _positive(beta, "beta")
        for name, value in (("classical", classical), ("saddle", saddle)):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
        eig, imaginary, rigid = self.spectrum(1 if saddle else 0)
        frequencies = np.sqrt(eig)
        log_electronic = math.log(self.electronic_degeneracy)
        p = None
        x = beta * frequencies
        if nbeads is not None:
            p = _integer(nbeads, "nbeads")
            if classical or saddle:
                raise ValueError("finite-P partition is only defined here for quantum reactants")
            # Product_n [(beta*omega/P)^2 + 4 sin^2(pi*n/P)]
            # = [2 sinh(P asinh(beta*omega/(2P)))]^2.
            x = 2 * p * np.arcsinh(x / (2 * p))
        log_vib = -float(np.log(x).sum() if classical else
                         (0.5*x + np.log(-np.expm1(-x))).sum())
        log_trans, log_rot = (_external_logs(self.geometry_bohr[None], self.masses,
                                             beta, self.symmetry_number)
                              if rigid else (0.0, 0.0))
        logq = log_vib + log_trans + log_rot + log_electronic
        return {"log_partition": float(logq), "log_vibrational_partition": log_vib,
                "log_translation_partition": log_trans, "log_rotation_partition": log_rot,
                "log_electronic_degeneracy": log_electronic,
                "positive_frequencies_au": frequencies.tolist(),
                "imaginary_frequency_au": imaginary,
                "zero_point_energy_hartree": float(0.5 * frequencies.sum()),
                "electronic_energy_hartree": self.energy_hartree,
                "removed_rigid_modes": rigid,
                "symmetry_number": self.symmetry_number,
                "electronic_degeneracy": self.electronic_degeneracy,
                "normalization": "per_bohr^3" if self.rigid_body else "dimensionless",
                "classical": bool(classical), "saddle": bool(saddle), "nbeads": p,
                "discretization": "primitive_harmonic" if p is not None else "continuum",
                "rigid_bead_powers_cancelled": bool(rigid and p is not None)}


def _compatible(geometry, masses, mask, rigid_body, reactants):
    if not reactants or not all(isinstance(s, HarmonicState) for s in reactants):
        raise ValueError("reactants must contain HarmonicState objects")
    combined = np.concatenate([s.masses for s in reactants])
    if combined.shape != np.asarray(masses).shape or not np.allclose(np.sort(combined), np.sort(masses), rtol=1e-12, atol=0):
        raise ValueError("reactant atom masses must match the reacting system, including isotopes")
    if any(s.rigid_body != rigid_body for s in reactants):
        raise ValueError("cannot mix gas and constrained partition conventions")
    if not rigid_body:
        if len(reactants) != 1 or not np.array_equal(reactants[0].active_mask, mask):
            raise ValueError("constrained models require one reactant with the same active_mask")
        if not np.allclose(reactants[0].masses, masses, rtol=1e-12, atol=0):
            raise ValueError("constrained models require the same atom ordering")
        if not np.allclose(reactants[0].geometry_bohr[~mask], geometry[~mask], rtol=0, atol=1e-12):
            raise ValueError("reactant frozen coordinates must match the reacting system")


def _rate_record(log_rate_au, molecularity, gas):
    power = 3 * (molecularity - 1) if gas else 0
    log_si = float(log_rate_au + power * math.log(BOHR_M) - math.log(ATOMIC_TIME_S))
    if not math.isfinite(log_si):
        raise InstantonError("nonfinite logarithmic rate")
    value = math.exp(log_si) if log_si <= math.log(sys.float_info.max) else None
    return {"rate": value, "rate_unit": "s^-1" if power == 0 else f"m^{power}/s",
            "log_rate_au": float(log_rate_au), "log_rate_si": log_si,
            "range_status": "overflow" if value is None else "underflow" if value == 0 else "finite"}


def _logsinh(x):
    return x + math.log(-math.expm1(-2*x)) - _LOG2 if x > 0 else -math.inf


def _logcosh(x):
    return abs(x) + math.log1p(math.exp(-2*abs(x))) - _LOG2


def _eckart_log_transmission(energy, forward, reverse, omega):
    threshold = max(0.0, forward - reverse)
    if energy <= threshold:
        return -math.inf
    a1, a2 = 2*math.pi*forward/omega, 2*math.pi*reverse/omega
    denominator = 1 / math.sqrt(a1) + 1 / math.sqrt(a2)
    a = 2 * math.sqrt(a1 * energy / forward) / denominator
    b = 2 * math.sqrt(2*math.pi/omega * (energy - forward + reverse)) / denominator
    d2 = a1*a2 - math.pi**2 / 4
    if d2 >= 0:
        logden = float(np.logaddexp(_logcosh(a+b), _logcosh(2*math.sqrt(d2))))
    else:
        # cosh(a+b)+cos(d) = 2*sinh((a+b)/2)^2+2*cos(d/2)^2.
        # This also avoids cancellation for very shallow barriers.
        logden = float(np.logaddexp(_LOG2 + 2*_logsinh((a+b)/2),
                                   _LOG2 + 2*math.log(abs(math.cos(math.sqrt(-d2))))))
    return min(0.0, _LOG2 + _logsinh(a) + _logsinh(b) - logden)


def eckart_transmission(energy_hartree, forward_barrier_hartree,
                        reverse_barrier_hartree, imaginary_frequency_au):
    """Exact asymmetric 1D Eckart transmission, including the channel threshold."""
    energy = float(energy_hartree)
    if not math.isfinite(energy):
        raise ValueError("energy_hartree must be finite")
    f = _positive(forward_barrier_hartree, "forward_barrier_hartree")
    r = _positive(reverse_barrier_hartree, "reverse_barrier_hartree")
    w = _positive(imaginary_frequency_au, "imaginary_frequency_au")
    return math.exp(_eckart_log_transmission(energy, f, r, w))


def eckart_correction(temperature_kelvin, forward_barrier_hartree,
                       reverse_barrier_hartree, imaginary_frequency_au):
    """Return log(kappa), with scaled adaptive quadrature, even at low T.

    kappa = beta*exp(beta*Vf)*integral[P(E)*exp(-beta*E), E_min, infinity].
    Both barrier heights are required. There is no implicit symmetric barrier.
    """
    beta = 1 / (KB_AU * _positive(temperature_kelvin, "temperature_kelvin"))
    f = _positive(forward_barrier_hartree, "forward_barrier_hartree")
    r = _positive(reverse_barrier_hartree, "reverse_barrier_hartree")
    w = _positive(imaginary_frequency_au, "imaginary_frequency_au")
    threshold = max(0.0, f-r)
    barrier = beta * min(f, r)
    upper = barrier + 60.0

    def log_integrand(u):
        return _eckart_log_transmission(threshold + u/beta, f, r, w) - u

    grid = np.unique(np.r_[np.geomspace(1e-10, upper, 160),
                           np.linspace(max(1e-10, barrier-30), upper, 60)])
    values = np.array([log_integrand(u) for u in grid])
    i = int(np.argmax(values))
    optimum = minimize_scalar(lambda u: -log_integrand(u), method="bounded",
                              bounds=(grid[max(i-1, 0)], grid[min(i+1, len(grid)-1)]))
    mode = float(optimum.x)
    peak = max(log_integrand(mode), float(values[i]))
    offsets = 2.0 ** np.arange(0, int(math.ceil(math.log2(upper))) + 1)
    points = np.unique(np.clip(np.r_[0, upper, mode, mode-offsets, mode+offsets, barrier], 0, upper))
    integral = error = 0.0
    for lo, hi in zip(points[:-1], points[1:]):
        part, err = quad(lambda u: math.exp(log_integrand(u)-peak), lo, hi,
                         epsabs=1e-11, epsrel=1e-9, limit=150)
        integral += part
        error += err
    if integral <= 0 or error / integral > 1e-6:
        raise InstantonError("Eckart quadrature did not converge")
    logk = barrier + peak + math.log(integral)
    return {"log_kappa": float(logk),
            "kappa": math.exp(logk) if logk <= math.log(sys.float_info.max) else None,
            "quadrature_relative_error": float(error/integral),
            "forward_barrier_hartree": f, "reverse_barrier_hartree": r}


def harmonic_cross_checks(temperature_kelvin, transition_state, reactants, *,
                          products=None, eckart_barriers_hartree=None,
                          tunnelling_baseline="harmonic_tst", reaction_degeneracy=1):
    """Classical/quantum harmonic TST and Wigner/Eckart estimates, without PES calls.

    Wigner and Eckart multiply the explicitly reported baseline. With products,
    Eckart heights include harmonic ZPE for harmonic_tst, but not classical_tst.
    Explicit (forward, reverse) heights override this choice and must use the
    caller's intended energy convention. Missing heights are reported, not fit
    from the local curvature. Gas rates use number density, not molar density.
    """
    temperature = _positive(temperature_kelvin, "temperature_kelvin")
    beta = 1 / (KB_AU * temperature)
    ts = transition_state
    _compatible(ts.geometry_bohr, ts.masses, ts.active_mask, ts.rigid_body, reactants)
    if tunnelling_baseline not in ("classical_tst", "harmonic_tst"):
        raise ValueError("tunnelling_baseline must be classical_tst or harmonic_tst")
    degeneracy = _integer(reaction_degeneracy, "reaction_degeneracy")
    positive, omega, _ = ts.spectrum(1)
    barrier = ts.energy_hartree - sum(s.energy_hartree for s in reactants)
    if barrier <= 0:
        raise InstantonError("positive forward electronic barrier required")
    report = {"temperature_kelvin": temperature, "tunnelling_baseline": tunnelling_baseline,
              "crossover_temperature_kelvin": omega / (2*math.pi*KB_AU),
              "imaginary_frequency_au": omega, "physical_converged": False,
              "warnings": ["harmonic wells and classical rigid rotors; not a recrossing calculation"]}
    for key, classical in (("classical_tst", True), ("harmonic_tst", False)):
        logq = ts.log_partition(beta, classical=classical, saddle=True)
        logq -= sum(s.log_partition(beta, classical=classical) for s in reactants)
        logk = math.log(degeneracy/(2*math.pi*beta)) + logq - beta*barrier
        report[key] = _rate_record(logk, len(reactants), ts.rigid_body)
    base = report[tunnelling_baseline]["log_rate_au"]
    log_wigner = math.log1p((beta*omega)**2 / 24)
    report["wigner"] = dict(_rate_record(base + log_wigner, len(reactants), ts.rigid_body),
                            kappa=math.exp(log_wigner), log_kappa=log_wigner)
    if beta*omega > 1:
        report["warnings"].append("Wigner is a weak-tunnelling expansion; beta*omega_b > 1, not a reliable low-T rate")
    convention = "explicit caller-supplied barriers"
    if eckart_barriers_hartree is None and products is not None:
        _compatible(ts.geometry_bohr, ts.masses, ts.active_mask, ts.rigid_body, products)
        zpe = lambda s, saddle=False: 0.5 * float(np.sqrt(s.spectrum(1 if saddle else 0)[0]).sum())
        quantum = tunnelling_baseline == "harmonic_tst"
        top = ts.energy_hartree + (0.5 * float(np.sqrt(positive).sum()) if quantum else 0)
        energies = [sum(s.energy_hartree + (zpe(s) if quantum else 0) for s in group)
                    for group in (reactants, products)]
        eckart_barriers_hartree = [top - e for e in energies]
        convention = "harmonic ZPE-corrected" if quantum else "electronic"
    if eckart_barriers_hartree is None:
        report["eckart"] = {"available": False, "reason": "forward/reverse barriers or product stationary-point data required"}
    else:
        barriers = np.asarray(eckart_barriers_hartree)
        if barriers.shape != (2,) or barriers.dtype.kind not in "fiu" or not np.isfinite(barriers).all():
            raise ValueError("eckart_barriers_hartree must contain two finite real heights")
        if np.any(barriers <= 0):
            report["eckart"] = {"available": False, "reason": "Eckart requires two positive barriers in the selected energy convention"}
        else:
            correction = eckart_correction(temperature, *barriers, omega)
            report["eckart"] = dict(_rate_record(base + correction["log_kappa"], len(reactants), ts.rigid_body),
                                    **correction, available=True, barrier_convention=convention)
    return report


def instanton_rate(action: RingPolymerAction, result, reactants, *, rigid_body=False,
                    symmetry_number=1, electronic_degeneracy=1, reaction_degeneracy=1,
                    action_hessian=None, gradient_tolerance=1e-8,
                    phase_tolerance=0.05):
    """Full Gaussian thermal RPI prefactor of an optimized multidimensional path.

    Reuse the Hessian cached by optimize_instanton(retain_hessian=True), pass
    an unweighted Cartesian *action* Hessian via action_hessian, or compute it
    once if neither is available. Remove only geometric rigid modes and the imaginary-time
    tangent. Retain the absolute value of the ONE negative mode. Reactants
    use the same primitive bead number. This is not uniform near crossover.
    """
    q, mask, mass = action._validate(result.path_bohr)
    p, d = len(q), len(mass)
    if p*d > 4096:
        raise ValueError("dense instanton prefactor is limited to 4096 active bead coordinates")
    if not isinstance(rigid_body, (bool, np.bool_)) or (rigid_body and not mask.all()):
        raise ValueError("rigid_body requires all Cartesian coordinates active")
    symmetry = _integer(symmetry_number, "symmetry_number")
    electronic = _integer(electronic_degeneracy, "electronic_degeneracy")
    degeneracy = _integer(reaction_degeneracy, "reaction_degeneracy")
    if not rigid_body and symmetry != 1:
        raise ValueError("constrained paths do not have a rotational symmetry number")
    tolerance = _positive(gradient_tolerance, "gradient_tolerance")
    phase_limit = _positive(phase_tolerance, "phase_tolerance")
    _compatible(q[0], action.masses, mask, rigid_body, reactants)
    if not result.converged or not math.isclose(result.temperature_kelvin, action.temperature_kelvin, rel_tol=1e-12):
        raise InstantonError("a converged orbit at the action temperature is required")
    evaluation = action.evaluate(q)
    if np.max(np.abs(evaluation["gradient"][:, mask])) > tolerance:
        raise InstantonError("path is not stationary for the supplied action")
    if not math.isclose(evaluation["action_hbar"], result.action_hbar, rel_tol=1e-10, abs_tol=1e-8):
        raise InstantonError("result and action use inconsistent PES or energy references")
    dtau = action.beta / p
    if action_hessian is None:
        action_hessian = getattr(result, "action_hessian", None)
    h = action.hessian(q) if action_hessian is None else _matrix(action_hessian, p*d, "action_hessian")
    repeated_mass = np.tile(mass, p)
    h = h / np.sqrt(repeated_mass[:, None] * repeated_mass[None, :])
    rigid = _rigid_basis(q, action.masses) if rigid_body else np.empty((p*d, 0))
    tangent = np.fft.ifft(2j*np.pi*np.fft.fftfreq(p)[:, None]
                          * np.fft.fft(q[:, mask], axis=0), axis=0).real
    tangent = (tangent*np.sqrt(mass)).ravel()
    tangent -= rigid @ (rigid.T @ tangent)
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-12 or evaluation["spring_action_hbar"] <= 0:
        raise InstantonError("collapsed path or no internal imaginary-time phase mode")
    tangent /= norm
    complement = _complement(np.column_stack((rigid, tangent)))
    eig = np.linalg.eigvalsh(complement.T @ h @ complement)
    scale = max(abs(float(eig[0])), 1e-30)
    _check_null(h, rigid, scale=scale)
    cutoff = scale * 1e-7
    if np.count_nonzero(eig < -cutoff) != 1 or np.any(np.abs(eig) <= cutoff):
        raise InstantonError("prefactor requires one negative mode and no unresolved internal zero modes")
    phase_residual = float(np.linalg.norm(h @ tangent) / scale)
    if phase_residual > phase_limit:
        raise InstantonError("imaginary-time phase mode is unresolved; increase beads and refine the orbit")
    log_vib = -0.5 * float(np.log(dtau * np.abs(eig)).sum())
    log_trans, log_rot = _external_logs(q, action.masses, dtau, symmetry) if rigid_body else (0.0, 0.0)
    # B=sum m*(q_j-q_{j-1})^2 = 2*dtau*S_spring.
    log_prefactor = (-math.log(dtau) + 0.5*math.log(evaluation["spring_action_hbar"]/math.pi)
                     + log_vib + log_trans + log_rot + math.log(electronic*degeneracy))
    reactant_diagnostics = [s.partition_diagnostics(action.beta, nbeads=p) for s in reactants]
    log_qr_internal = sum(s["log_partition"] for s in reactant_diagnostics)
    reactant_energy = sum(s.energy_hartree for s in reactants)
    log_qr_electronic = -action.beta * (reactant_energy - action.energy_reference)
    log_qr = log_qr_internal + log_qr_electronic
    log_rate = log_prefactor - evaluation["action_hbar"] - log_qr
    rigid_residuals = (np.linalg.norm(h @ rigid, axis=0) / scale).tolist()
    return dict(_rate_record(log_rate, len(reactants), rigid_body),
                method="multidimensional_thermal_ring_polymer_instanton",
                temperature_kelvin=action.temperature_kelvin, nbeads=p,
                log_fluctuation_partition=log_vib, log_translation_partition=log_trans,
                log_rotation_partition=log_rot, log_prefactor_au=log_prefactor,
                log_reactant_partition=log_qr, removed_rigid_modes=rigid.shape[1],
                reactant_partitions=reactant_diagnostics,
                log_reactant_electronic_boltzmann=log_qr_electronic,
                energy_reference_hartree=action.energy_reference,
                reactant_energy_hartree=reactant_energy,
                removed_translation_modes=3 if rigid_body else 0,
                removed_rotation_modes=rigid.shape[1] - 3 if rigid_body else 0,
                removed_phase_modes=1,
                rigid_mode_relative_residuals=rigid_residuals,
                mass_weighted_action_negative_eigenvalue=float(eig[0]),
                positive_fluctuation_modes=int(np.count_nonzero(eig > cutoff)),
                reaction_degeneracy=degeneracy, electronic_degeneracy=electronic,
                symmetry_number=symmetry, molecularity=len(reactants),
                partition_convention=("isolated_gas_number_density_classical_rigid_rotors"
                                      if rigid_body else "bound_or_constrained_cartesian"),
                phase_mode_relative_residual=phase_residual, negative_modes=1,
                numerical_converged=True, physical_converged=False,
                limitations=["semiclassical; nonuniform near crossover",
                             "bead, Hessian-step, PES and reactant-partition convergence required",
                             "classical rigid rotors for isolated gas molecules; no hindered rotors"])


def print_cross_checks(report, *, file=None):
    """Print the same explicitly labelled estimates that are returned as JSON."""
    stream = sys.stdout if file is None else file
    print(f"Low-temperature cross-checks at {report['temperature_kelvin']:.8g} K", file=stream)
    print(f"Wigner/Eckart baseline: {report['tunnelling_baseline']}", file=stream)
    for key in ("classical_tst", "harmonic_tst", "wigner", "eckart"):
        row = report[key]
        if row.get("available") is False:
            print(f"{key}: unavailable ({row['reason']})", file=stream)
        else:
            value = "overflow" if row["rate"] is None else f"{row['rate']:.10e}"
            print(f"{key}: {value} {row['rate_unit']} (ln k_SI={row['log_rate_si']:.10g})", file=stream)
    for warning in report["warnings"]:
        print(f"WARNING: {warning}", file=stream)


def main(argv=None):
    """python -m pyrpmd.low_temperature stationary_points.json [--json]."""
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Cached stationary-point energies and Cartesian Hessians")
    parser.add_argument("--json", action="store_true", help="Print machine-readable results instead of the table")
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.input.read_text())
        ts = HarmonicState(**data.pop("transition_state"))
        reactants = [HarmonicState(**item) for item in data.pop("reactants")]
        if "products" in data:
            data["products"] = [HarmonicState(**item) for item in data["products"]]
        report = harmonic_cross_checks(transition_state=ts, reactants=reactants, **data)
        if args.json:
            print(json.dumps(report, indent=2, allow_nan=False))
        else:
            print_cross_checks(report)
    except (ValueError, KeyError, TypeError, OSError, InstantonError) as exc:
        parser.exit(2, f"low-temperature cross-checks: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
