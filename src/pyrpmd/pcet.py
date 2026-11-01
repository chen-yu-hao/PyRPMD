"""Local, vibronically nonadiabatic electrochemical PCET model rates.

The transferring nucleus is solved on a one-dimensional Dirichlet grid.  The
remaining environment is a classical Gaussian reorganization bath; a wide-band
electrode supplies/accepts one electron.  These are golden-rule model rates,
not RPMD trajectories or a constant-potential atomistic simulation.

Conventions follow the electrode extension of the vibronic golden rule in
Venkataraman et al., J. Phys. Chem. C 112, 12386 (2008),
doi:10.1021/jp802171y, and Hutchison et al., J. Chem. Phys. 163, 091501 (2025),
doi:10.1063/5.0284337.  This implementation is independent of their software.

All energies/coordinates/masses are atomic units.  Temperature is kelvin and
returned rates are s^-1.  ``driving_force_au`` is the reduced-minus-oxidized
well-minimum free-energy offset, EXCLUDING this proton's zero-point energy.
Electronic-state curves are separately referenced; their solved proton energies
supply zero-point and excited-state energies exactly once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import eigh_tridiagonal
from scipy.special import expit, logsumexp

from .constants import AMU_TO_AU, ATOMIC_TIME_S, KB_AU
from .validation import finite_real

# Match the isotope mass convention used by System (neutral atomic masses).
ISOTOPE_MASSES_AU = {
    "H": 1.00782503223 * AMU_TO_AU,
    "D": 2.01410177812 * AMU_TO_AU,
    "T": 3.01604928199 * AMU_TO_AU,
}


_real = partial(finite_real, coerce=True, shape='scalar')


def _array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return np.asarray(array, dtype=float)


@dataclass(frozen=True)
class ProtonSpectrum:
    """Finite-difference spectrum with discrete-normalized eigenvector columns.

    ``wavefunctions[:, i] / sqrt(dx)`` is the coordinate-space wavefunction.
    Eigenvectors include the two zero Dirichlet endpoint values.  Overlaps on a
    common grid are consequently ordinary matrix products, without another dx.
    ``boundary_probabilities`` measures the outer 5% of *interior* grid points
    at each end, rather than the endpoints which vanish by construction.
    """

    grid_au: np.ndarray
    potential_au: np.ndarray
    mass_au: float
    energies_au: np.ndarray
    wavefunctions: np.ndarray
    energy_origin_au: float
    boundary_probabilities: np.ndarray


def solve_proton(grid_au, potential_au, mass_au, nstates, *, energy_origin_au=None):
    """Solve -1/(2m) d²/dx² + V by second-order finite differences.

    The grid includes both Dirichlet boundaries.  Supply the known physical
    minimum as ``energy_origin_au`` for an analytic curve (zero for the usual
    harmonic form); otherwise the sampled potential minimum is subtracted.
    A tabulated model should supply its original table's minimum so changing
    interpolation grids cannot silently change the energy reference.
    """
    grid = _array(grid_au, "proton grid")
    potential = _array(potential_au, "proton potential")
    if grid.ndim != 1 or grid.size < 5:
        raise ValueError("proton grid must be one-dimensional with at least 5 points")
    if potential.shape != grid.shape:
        raise ValueError("proton potential must have the same shape as its grid")
    spacing = np.diff(grid)
    if not np.all(spacing > 0) or not np.allclose(
        spacing, spacing[0], rtol=1e-10, atol=abs(spacing[0]) * 1e-10
    ):
        raise ValueError("proton grid must be strictly increasing and uniformly spaced")
    if (
        isinstance(nstates, (bool, np.bool_))
        or not isinstance(nstates, (int, np.integer))
        or not 1 <= nstates <= grid.size - 2
    ):
        raise ValueError("proton nstates must be an integer between 1 and grid_points - 2")
    mass = _real(mass_au, "proton mass", positive=True)
    origin = (
        float(np.min(potential))
        if energy_origin_au is None
        else _real(energy_origin_au, "proton energy origin")
    )
    shifted = potential - origin
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        kinetic = 1.0 / (mass * spacing[0] ** 2)
    if not np.isfinite(kinetic) or not np.isfinite(shifted).all():
        raise ValueError("proton grid, mass or energy scale exceeds numerical range")
    diagonal = shifted[1:-1] + kinetic
    if not np.isfinite(diagonal).all():
        raise ValueError("proton Hamiltonian exceeds numerical range")
    energies, vectors = eigh_tridiagonal(
        diagonal,
        np.full(grid.size - 3, -0.5 * kinetic),
        select="i",
        select_range=(0, int(nstates) - 1),
        check_finite=False,
    )
    wavefunctions = np.zeros((grid.size, int(nstates)))
    wavefunctions[1:-1] = vectors
    edge = max(1, int(math.ceil((grid.size - 2) * 0.05)))
    boundary = np.sum(vectors[:edge] ** 2, axis=0) + np.sum(vectors[-edge:] ** 2, axis=0)
    return ProtonSpectrum(
        grid.copy(), shifted.copy(), mass, energies, wavefunctions, origin, boundary
    )


def electrode_log_kernels(
    gaps_au, *, temperature_K, reorganization_au, rtol=1e-9, atol=1e-11
):
    """Log wide-band Fermi/Gaussian kernels for paired red/ox channels.

    For g = G_reduced - G_oxidized including proton state energies,

      A(g) = integral f(eps) exp[-(lambda+g-eps)^2/(4 lambda kT)]
                       / sqrt(4 pi lambda kT) d eps.

    The paired oxidation kernel is A(-g).  It also equals the integral with
    1-f(eps) and the reverse Gaussian, in the same electrode energy reference.
    A(g)/A(-g)=exp(-g/kT).  We evaluate the favorable direction independently
    in log scale and use this identity for the exponentially small direction.
    This retains meaningful log rates when ordinary floating-point rates
    underflow.

    The Gaussian variable is shifted to its unique log-integrand maximum.
    Its second derivative is <= -2, bounding the discarded |y|>10 tails by
    exp(log_peak)*sqrt(pi)*erfc(10).  Breakpoints around the minimum possible
    peak width prevent adaptive quadrature from missing a very narrow mode.
    """
    gaps = _array(gaps_au, "PCET channel gaps")
    if gaps.size == 0:
        raise ValueError("PCET channel gaps must not be empty")
    temperature = _real(temperature_K, "temperature", positive=True)
    reorganization = _real(reorganization_au, "reorganization energy", positive=True)
    rtol = _real(rtol, "quadrature rtol", positive=True)
    atol = _real(atol, "quadrature atol", positive=True)
    if rtol >= 1 or atol >= 1:
        raise ValueError("quadrature rtol and atol must be smaller than one")
    thermal = KB_AU * temperature
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratio = reorganization / thermal
        favorable = -np.abs(gaps.ravel())
        a = 2 * np.sqrt(ratio)
        b = (reorganization + favorable) / thermal
        scaled_gaps = gaps.ravel() / thermal
    if thermal <= 0 or not np.isfinite(a) or not np.isfinite(b).all() or not np.isfinite(scaled_gaps).all():
        raise ValueError("PCET energy/temperature ratio exceeds numerical range")
    # The derivative -2*x - a*expit(b+a*x) is monotone decreasing.
    lower = np.full(b.shape, -a / 2)
    upper = np.zeros(b.shape)
    for _ in range(64):
        midpoint = (lower + upper) / 2
        derivative = -2 * midpoint - a * expit(b + a * midpoint)
        lower = np.where(derivative > 0, midpoint, lower)
        upper = np.where(derivative > 0, upper, midpoint)
    mode = (lower + upper) / 2
    peak = -mode * mode - np.logaddexp(0, b + a * mode)
    if not np.isfinite(peak).all():
        raise ValueError("PCET log kernel exceeds numerical range")

    def integrand(y):
        x = mode + y
        log_value = -x * x - np.logaddexp(0, b + a * x)
        return np.exp(np.minimum(0.0, log_value - peak))

    width = 1 / math.sqrt(1 + ratio / 2)
    points = sorted({0.0, -1.0, 1.0, -width, width, -10 * width, 10 * width})
    points = [point for point in points if -10 < point < 10]
    integrals, error, info = quad_vec(
        integrand, -10.0, 10.0, epsabs=atol, epsrel=rtol, norm="max",
        points=points, limit=2000, full_output=True,
    )
    integrals = np.asarray(integrals)
    if not info.success or not np.isfinite(integrals).all() or np.any(integrals <= 0):
        raise RuntimeError("PCET electrode quadrature did not converge; check energy scales and tolerances")
    favorable_log = peak + np.log(integrals / math.sqrt(math.pi))
    # The integral is an expectation of a Fermi function and cannot exceed 1.
    if np.any(favorable_log > 10 * max(rtol, atol)):
        raise RuntimeError("PCET electrode quadrature produced a kernel larger than one")
    favorable_log = np.minimum(favorable_log, 0.0)
    red = favorable_log - np.maximum(scaled_gaps, 0.0)
    ox = favorable_log + np.minimum(scaled_gaps, 0.0)
    tail_bound = math.sqrt(math.pi) * math.erfc(10)
    diagnostics = {
        "method": "adaptive_log_scaled_fermi_gaussian",
        "success": True,
        "requested_rtol": rtol,
        "requested_atol": atol,
        "scaled_integral_absolute_error_estimate": float(error),
        "max_kernel_relative_error_estimate": float((error + tail_bound) / np.min(integrals)),
        "scaled_integral_tail_bound": tail_bound,
        "evaluations": int(info.neval),
    }
    return red.reshape(gaps.shape), ox.reshape(gaps.shape), diagnostics


def _rate_components(log_terms):
    total_log = float(logsumexp(log_terms))
    if total_log == -math.inf:
        return 0.0, None, np.zeros_like(log_terms), np.zeros_like(log_terms)
    if not math.isfinite(total_log) or total_log > math.log(np.finfo(float).max):
        raise ValueError("PCET rate exceeds floating-point range; check coupling and DOS units")
    with np.errstate(under="ignore"):
        contributions = np.exp(log_terms)
        fractions = np.exp(log_terms - total_log)
        rate = float(np.exp(total_log))
    return rate, total_log, contributions, fractions


def golden_rule_rates(
    oxidized,
    reduced,
    *,
    temperature_K,
    driving_force_au,
    reorganization_au,
    coupling_au,
    density_per_hartree,
    quadrature_rtol=1e-9,
    quadrature_atol=1e-11,
):
    """Compute local first-order red/ox rates on the same proton basis grid.

    ``density_per_hartree`` includes the user's chosen spin/channel degeneracy
    once.  The prefactor is 2*pi*rho*|V_el|² / atomic_time.  Vibronic coupling
    is V_el times the proton overlap, so this backend additionally assumes
    electronically nonadiabatic, coordinate-independent electronic coupling.

    The O and R potential curves each have their own energy origin.  Their
    relative well-minimum free energy enters ONLY through ``driving_force_au``.
    For a one-electron reduction its potential shift is +e*(U-U_ref).
    """
    if not isinstance(oxidized, ProtonSpectrum) or not isinstance(reduced, ProtonSpectrum):
        raise ValueError("PCET rates require two ProtonSpectrum instances")
    if not np.array_equal(oxidized.grid_au, reduced.grid_au):
        raise ValueError("PCET proton spectra must use the same coordinate grid")
    if oxidized.mass_au != reduced.mass_au:
        raise ValueError("PCET proton spectra must use the same isotope mass")
    temperature = _real(temperature_K, "temperature", positive=True)
    driving = _real(driving_force_au, "driving force")
    coupling = _real(coupling_au, "electronic coupling", nonnegative=True)
    density = _real(density_per_hartree, "electrode density of states", nonnegative=True)
    reorganization = _real(reorganization_au, "reorganization energy", positive=True)
    thermal = KB_AU * temperature
    if thermal <= 0 or not math.isfinite(thermal):
        raise ValueError("PCET temperature exceeds numerical range")
    eo = oxidized.energies_au
    er = reduced.energies_au
    log_zo = float(logsumexp(-eo / thermal))
    log_zr = float(logsumexp(-er / thermal))
    log_po = -eo / thermal - log_zo
    log_pr = -er / thermal - log_zr
    overlaps = oxidized.wavefunctions.T @ reduced.wavefunctions
    overlap_squared = np.clip(overlaps * overlaps, 0.0, 1.0)
    gaps = driving + er[None, :] - eo[:, None]
    log_red_kernel, log_ox_kernel, quadrature = electrode_log_kernels(
        gaps, temperature_K=temperature, reorganization_au=reorganization,
        rtol=quadrature_rtol, atol=quadrature_atol,
    )
    with np.errstate(divide="ignore"):
        log_overlaps = np.log(overlap_squared)
    if coupling == 0 or density == 0:
        log_prefactor = -math.inf
    else:
        log_prefactor = (
            math.log(2 * math.pi) + math.log(density)
            + 2 * math.log(coupling) - math.log(ATOMIC_TIME_S)
        )
    red = _rate_components(log_prefactor + log_po[:, None] + log_overlaps + log_red_kernel)
    ox = _rate_components(log_prefactor + log_pr[None, :] + log_overlaps + log_ox_kernel)
    proton_delta_f = -thermal * (log_zr - log_zo)
    reaction_delta_f = driving + proton_delta_f
    expected_log_ratio = -reaction_delta_f / thermal
    residual = (
        red[1] - ox[1] - expected_log_ratio
        if red[1] is not None and ox[1] is not None else None
    )
    return {
        "method": "electrochemical_vibronic_golden_rule",
        "normalization": "local_first_order",
        "rate_units": "s^-1",
        "reduction_rate_s": red[0],
        "oxidation_rate_s": ox[0],
        "log_reduction_rate_s": red[1],
        "log_oxidation_rate_s": ox[1],
        "reduction_contributions_s": red[2],
        "oxidation_contributions_s": ox[2],
        "reduction_channel_fractions": red[3],
        "oxidation_channel_fractions": ox[3],
        "oxidized_population": np.exp(log_po),
        "reduced_population": np.exp(log_pr),
        "proton_partition_log_oxidized": log_zo,
        "proton_partition_log_reduced": log_zr,
        "proton_free_energy_difference_au": proton_delta_f,
        "reaction_free_energy_au": reaction_delta_f,
        "expected_log_rate_ratio": expected_log_ratio,
        "detailed_balance_log_residual": residual,
        "overlap_capture_oxidized": float(np.exp(log_po) @ np.sum(overlap_squared, axis=1)),
        "overlap_capture_reduced": float(np.exp(log_pr) @ np.sum(overlap_squared, axis=0)),
        "quadrature": quadrature,
    }
