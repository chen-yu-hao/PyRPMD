"""Deterministic, positive-diffusion GLE fits for harmonic reference systems.

This implementation generates its matrices from equations, never from a
third-party parameter table.  Public unit conversion and file handling live in
``gle_parameters``.  The matrices returned here use atomic time and covariance
normalized by the ring-polymer temperature ``P / beta``.
"""
from __future__ import annotations

import math
import numbers

import numpy as np
import scipy
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.optimize import least_squares

_SCHEMA = "pyrpmd.gle-parameters.v1"
_VERSION = "positive-arrowhead-harmonic-v1"
_PIGLET_VERSION = "positive-arrowhead-harmonic-relaxation-v2"
_SOLVER_TOLERANCES = {"ftol": 1e-9, "xtol": 1e-9, "gtol": 1e-7}
_PIGLET_RATE_FLOOR = .05
_PIGLET_DECAY_TARGET = .3
_PIGLET_MINIMUM_DECAY = .27
_PIGLET_PROBE_TIME = 12.5
_PIGLET_MAX_TRANSIENT_ERROR = .005


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a positive finite real number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real number")
    return value


def _integer(value, name, minimum, maximum=None):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, numbers.Integral)
            or value < minimum or maximum is not None and value > maximum):
        upper = "" if maximum is None else f" and at most {maximum}"
        raise ValueError(f"{name} must be an integer at least {minimum}{upper}")
    return int(value)


def _settings(frequency_range_au, naux, grid_size, max_nfev):
    raw = np.asarray(frequency_range_au, dtype=object)
    if raw.shape != (2,):
        raise ValueError("frequency_range_au must contain two finite positive, strictly increasing frequencies")
    band = np.array([_real(x, "frequency_range_au") for x in raw], dtype=np.float64)
    if band[0] >= band[1]:
        raise ValueError("frequency_range_au must contain two finite positive, strictly increasing frequencies")
    naux = _integer(naux, "naux", 1, 3)
    grid_size = _integer(grid_size, "grid_size", 8, 256)
    max_nfev = _integer(max_nfev, "max_nfev", 1)
    return band, naux, grid_size, max_nfev


def _arrowhead(log_parameters, size, *, thermal):
    """Positive symmetric part by construction, for both supported families."""
    rates = np.exp(log_parameters[:size])
    couplings = np.exp(log_parameters[size:2 * size - 1])
    drift = np.diag(rates)
    drift[0, 1:] = couplings
    if thermal:
        drift[1:, 0] = -couplings
        temperatures = np.exp(log_parameters[2 * size - 1:])
        diffusion = np.diag(2 * rates * temperatures)
    else:
        # The Schur complement is rates[0] > 0.  This symmetric coupling gives
        # friction that increases from low to high frequency, unlike a purely
        # skew arrowhead.  The canonical covariance remains exactly I.
        drift[1:, 0] = couplings
        drift[0, 0] += np.sum(couplings**2 / rates[1:])
        diffusion = 2 * drift
    return drift, diffusion


def _oscillator_drift(drift, frequencies):
    size = len(drift) + 1
    result = np.zeros((len(frequencies), size, size))
    result[:, 0, 1] = -1.0
    result[:, 1, 0] = np.square(frequencies)
    result[:, 1:, 1:] = drift
    return result


def _lyapunov_00(drifts, diffusion):
    """Batched tiny solves; used only during a fit, never during propagation."""
    count, size, _ = drifts.shape
    identity = np.eye(size)
    # Row-major vec(A C + C A.T), with (i,j) and (k,l) paired.
    operator = (np.einsum("nik,jl->nijkl", drifts, identity)
                + np.einsum("ik,njl->nijkl", identity, drifts))
    operator = operator.reshape(count, size * size, size * size)
    right = np.broadcast_to(diffusion.ravel(), (count, size * size))
    solution = np.linalg.solve(operator, right[..., None])[..., 0]
    return solution[:, 0]


def _mode_labels(beads):
    return ([0] + [k for k in range(1, (beads + 1) // 2) for _ in range(2)]
            + ([beads // 2] if beads % 2 == 0 else []))


def _ring_frequencies(beads):
    labels, weights = np.unique(_mode_labels(beads)[1:], return_counts=True)
    return 2 * beads * np.sin(np.pi * labels / beads), weights


def _piglet_errors(drift, diffusion, dimensionless_frequencies, beads, *, independent=False):
    """Relative V and K_cv errors; the classical centroid makes them equal."""
    springs, weights = _ring_frequencies(beads)
    modes = np.sqrt(np.square(dimensionless_frequencies[:, None]) + springs**2)
    drifts = _oscillator_drift(drift, modes.ravel())
    right = np.zeros((len(drift) + 1, len(drift) + 1))
    right[1:, 1:] = diffusion
    if independent:
        widths = np.array([solve_continuous_lyapunov(a, right)[0, 0] for a in drifts])
    else:
        widths = _lyapunov_00(drifts, right)
    widths = widths.reshape(modes.shape)
    potential = .5 + .5 * dimensionless_frequencies**2 * (widths @ weights)
    exact = dimensionless_frequencies / (4 * np.tanh(dimensionless_frequencies / 2))
    errors = potential / exact - 1.0
    if not np.isfinite(errors).all() or np.any(widths <= 0):
        raise ValueError("PIGLET harmonic covariance solve failed; reduce the frequency range")
    return errors


def _solver_metadata(fit, initial, lower, upper, max_nfev, *, tolerances=None):
    return {
        "name": "scipy.optimize.least_squares",
        "method": "trf", "jacobian": "2-point", "loss": "linear",
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "initial_log_parameters": initial.tolist(),
        "log_parameter_lower_bounds": lower.tolist(),
        "log_parameter_upper_bounds": upper.tolist(),
        "max_nfev": max_nfev,
        "tolerances": dict(_SOLVER_TOLERANCES if tolerances is None else tolerances),
        "success": bool(fit.success), "status": int(fit.status),
        "message": str(fit.message), "nfev": int(fit.nfev),
        "optimality": float(fit.optimality), "cost": float(fit.cost),
    }


def _validate_covariance(drift, diffusion):
    covariance = solve_continuous_lyapunov(drift, diffusion)
    covariance = .5 * (covariance + covariance.T)
    residual = drift @ covariance + covariance @ drift.T - diffusion
    relative = float(np.linalg.norm(residual) / np.linalg.norm(diffusion))
    if (not np.isfinite(covariance).all()
            or np.linalg.eigvalsh(covariance).min() <= 0
            or np.linalg.eigvalsh(diffusion).min() <= 0
            or relative > 1e-9):
        raise ValueError("generated GLE covariance failed positive-definiteness or Lyapunov validation")
    return covariance, relative


def _internal_decays(drift, frequencies, beads):
    springs, _ = _ring_frequencies(beads)
    frequencies = np.sqrt(frequencies[:, None]**2 + springs**2)
    drifts = _oscillator_drift(drift, frequencies.ravel())
    return np.linalg.eigvals(drifts).real.min(axis=1).reshape(frequencies.shape)


def _relaxation_reference(drift, diffusion, covariance, frequencies, beads):
    """Independent full-SDE transients for two explicitly stated initial laws."""
    springs, _ = _ring_frequencies(beads)
    mode_frequencies = np.sqrt(frequencies[:, None]**2 + springs**2)
    drifts = _oscillator_drift(drift, mode_frequencies.ravel())
    right = np.zeros_like(drifts[0])
    right[1:, 1:] = diffusion
    free_initial = np.zeros_like(right)
    free_initial[1:, 1:] = covariance
    minimum_decay = math.inf
    maximum_free_error = maximum_cold_error = maximum_tau = 0.0
    for full_drift in drifts:
        stationary = solve_continuous_lyapunov(full_drift, right)
        transition = expm(-_PIGLET_PROBE_TIME * full_drift)
        free_error = (transition @ (free_initial - stationary) @ transition.T)[0, 0]
        cold_error = (transition @ stationary @ transition.T)[0, 0]
        maximum_free_error = max(maximum_free_error, abs(float(free_error / stationary[0, 0])))
        maximum_cold_error = max(maximum_cold_error, abs(float(cold_error / stationary[0, 0])))
        minimum_decay = min(minimum_decay, float(np.linalg.eigvals(full_drift).real.min()))
        correlation = stationary[:, 0] / stationary[0, 0]
        tau = solve_continuous_lyapunov(full_drift, np.outer(correlation, correlation))[0, 0]
        maximum_tau = max(maximum_tau, float(tau))
    return {
        "minimum_decay_beta": minimum_decay,
        "maximum_free_bath_relative_deviation": maximum_free_error,
        "maximum_cold_relative_deviation": maximum_cold_error,
        "maximum_tau_V_over_beta": maximum_tau,
    }


def fit_piglet(frequency_range_au, *, beta, beads, naux=2, grid_size=32,
               max_nfev=600, tolerance=.01):
    """Fit PIGLET harmonic potential and centroid-virial kinetic energies.

    Internal real ring modes share an arrowhead GLE.  A strictly positive
    diagonal diffusion and stable drift are fitted together; their free-bath
    covariance is then obtained by a Lyapunov solve.  The centroid uses an
    independent canonical bath.  A successful optimizer and a dense independent
    harmonic check within ``tolerance`` are both mandatory.  A full harmonic
    relaxation penalty and independent transient acceptance checks prevent
    fitting static moments with a practically unequilibrated auxiliary mode.
    These fixed controls are implementation choices recorded in metadata,
    rather than a prescribed parameter set from the PIGLET paper.  This is a local
    fit with one documented deterministic initial condition, not a guarantee of
    a global optimum or of accuracy for anharmonic systems.
    """
    band, naux, grid_size, max_nfev = _settings(
        frequency_range_au, naux, grid_size, max_nfev)
    beta = _real(beta, "beta")
    beads = _integer(beads, "beads", 2)
    tolerance = _real(tolerance, "tolerance")
    if tolerance >= 1:
        raise ValueError("tolerance must be less than one")
    dimensionless_band = band * beta
    if not np.isfinite(dimensionless_band).all() or np.any(dimensionless_band <= 0):
        raise ValueError("beta times frequency_range_au must be finite and positive")
    grid = np.geomspace(*dimensionless_band, grid_size)
    heldout = np.geomspace(*dimensionless_band, 8 * grid_size + 1)
    size = naux + 1
    scale = max(float(beads), float(grid[-1]))
    if naux == 1:
        rates = np.array([beads, 2 * beads], dtype=float)
        couplings = np.array([2.5 * beads], dtype=float)
    else:
        rates = np.r_[beads, np.geomspace(.1 * beads, 2 * beads, naux)]
        couplings = np.geomspace(beads, 4 * beads, naux)
    temperatures = np.r_[max(4., 2 * grid[-1] / beads), np.full(naux, .2)]
    initial = np.log(np.r_[rates, couplings, temperatures])
    lower = np.log(np.r_[np.full(size, scale * _PIGLET_RATE_FLOOR),
                         np.full(size - 1, scale * 1e-6), np.full(size, 1e-6)])
    upper = np.log(np.r_[np.full(2 * size - 1, scale * 100), np.full(size, max(1000., scale * 100))])
    # This is the documented deterministic initializer, not a fit retry.
    initial = np.maximum(initial, lower + math.log(1.01))
    if np.any(initial >= upper):
        raise ValueError("PIGLET frequency range is outside the documented fitting parameter bounds")
    solver_tolerances = {"ftol": 1e-8, "xtol": 1e-8,
                         "gtol": min(1e-5, max(1e-12, .1 * tolerance**2))}

    def residual(parameters):
        a, d = _arrowhead(parameters, size, thermal=True)
        errors = _piglet_errors(a, d, grid, beads)
        decays = _internal_decays(a, grid, beads)
        # Equal per-frequency weight independently of the number of modes.
        penalty = np.maximum(0., 1. - decays / _PIGLET_DECAY_TARGET) / math.sqrt(decays.shape[1])
        return np.r_[errors, penalty.ravel()]

    fit = least_squares(residual, initial, bounds=(lower, upper),
                        max_nfev=max_nfev, **solver_tolerances)
    drift, diffusion = _arrowhead(fit.x, size, thermal=True)
    train_errors = _piglet_errors(drift, diffusion, grid, beads)
    # Independent Bartels-Stewart solves assess frequencies outside the fit grid.
    check_errors = _piglet_errors(drift, diffusion, heldout, beads, independent=True)
    maximum = float(np.max(np.abs(check_errors)))
    if not fit.success or maximum > tolerance:
        raise ValueError(
            f"PIGLET fit did not converge: optimizer_success={fit.success}, "
            f"maximum_relative_error={maximum:.9g}, tolerance={tolerance:.9g}; "
            "adjust naux, frequency range, or max_nfev explicitly")
    covariance, lyapunov_residual = _validate_covariance(drift, diffusion)
    relaxation = _relaxation_reference(drift, diffusion, covariance, heldout, beads)
    maximum_transient = max(relaxation["maximum_free_bath_relative_deviation"],
                            relaxation["maximum_cold_relative_deviation"])
    if (not all(math.isfinite(value) for value in relaxation.values())
            or relaxation["minimum_decay_beta"] < _PIGLET_MINIMUM_DECAY
            or maximum_transient > _PIGLET_MAX_TRANSIENT_ERROR):
        raise ValueError(
            "PIGLET fit failed internal harmonic relaxation validation: "
            f"minimum_decay_beta={relaxation['minimum_decay_beta']:.9g}, "
            f"maximum_transient_relative_error={maximum_transient:.9g}; "
            "adjust naux or the frequency range explicitly")
    physical_drift = drift / beta
    centroid_rate = (math.sqrt(float(dimensionless_band[0]))
                     * math.sqrt(float(dimensionless_band[1])) / beta)
    matrices_a = np.repeat(physical_drift[None], beads, axis=0)
    matrices_c = np.repeat(covariance[None], beads, axis=0)
    matrices_a[0] = np.eye(size) * centroid_rate
    matrices_c[0] = np.eye(size)
    if not np.isfinite(matrices_a).all() or np.min(np.diag(physical_drift)) <= 0:
        raise ValueError("generated drift cannot be represented in atomic units")
    metadata = {
        "schema": _SCHEMA, "generator": "piglet-fit", "implementation": _PIGLET_VERSION,
        "frequency_range_au": band.tolist(), "frequency_grid_au": (grid / beta).tolist(),
        "validation_frequency_grid_au": (heldout / beta).tolist(),
        "frequency_convention": "angular", "beta": beta, "beads": beads,
        "naux": naux, "grid_size": grid_size,
        "objective": "relative harmonic potential and centroid-virial kinetic energy error plus internal harmonic relaxation penalty",
        "target": "omega/(4*tanh(beta*omega/2)) per Cartesian oscillator",
        "parameterization": "A=positive_diagonal+arrowhead_skew; D=positive_diagonal; A*C+C*A.T=D",
        "normal_mode_order": _mode_labels(beads),
        "covariance_normalization": "dimensionless: divide mass-scaled covariance by P/beta",
        "training_relative_errors": train_errors.tolist(),
        "training_internal_decay_rates_beta": _internal_decays(drift, grid, beads).tolist(),
        "validation_relative_errors": check_errors.tolist(),
        "maximum_relative_error": maximum,
        "rms_relative_error": float(np.sqrt(np.mean(check_errors**2))),
        "lyapunov_relative_residual": lyapunov_residual,
        "tolerance": tolerance, "fit_converged": True, "physical_converged": False,
        "scope": "harmonic moments on the declared band; no anharmonic, timestep, sampling or NPT convergence claim",
        "relaxation_controls": {
            "origin": "fixed PyRPMD fitting controls, not a published PIGLET parameter prescription",
            "rate_lower_bound_over_max_beads_beta_omega": _PIGLET_RATE_FLOOR,
            "penalty_target_decay_beta": _PIGLET_DECAY_TARGET,
            "penalty": "max(0,1-beta*decay/target)/sqrt(number_of_distinct_internal_modes)",
            "minimum_accepted_decay_beta": _PIGLET_MINIMUM_DECAY,
            "probe_time_over_beta": _PIGLET_PROBE_TIME,
            "probe_time_au": _PIGLET_PROBE_TIME * beta,
            "maximum_accepted_transient_relative_error": _PIGLET_MAX_TRANSIENT_ERROR,
            "free_bath_initial_covariance": "block_diag(0,C): q=0; bath at its stationary free-particle covariance; no q-bath correlations",
            "cold_initial_covariance": "zero covariance of all q,p,s variables",
            "scope": "internal harmonic modes only; not arbitrary initial laws, the centroid, or anharmonic systems",
        },
        "relaxation_diagnostics": {
            **relaxation,
            "minimum_decay_rate_au": relaxation["minimum_decay_beta"] / beta,
            "maximum_tau_V_au": relaxation["maximum_tau_V_over_beta"] * beta,
        },
        "solver": _solver_metadata(fit, initial, lower, upper, max_nfev,
                                    tolerances=solver_tolerances),
    }
    return {
        "A": matrices_a.tolist(), "C": matrices_c.tolist(), "beta": beta, "beads": beads,
        "provenance": f"PyRPMD generated {metadata['generator']} {_PIGLET_VERSION}; no third-party matrix data",
        "parameter_metadata": metadata,
    }


def _potential_tau(drift, frequencies, *, independent=False):
    """Integral of squared normalized q autocorrelation of a canonical GLE."""
    drifts = _oscillator_drift(drift, frequencies)
    right = np.zeros((len(drift) + 1, len(drift) + 1))
    right[0, 0] = 1.0
    if independent:
        result = np.array([solve_continuous_lyapunov(a, right)[0, 0] for a in drifts])
    else:
        result = _lyapunov_00(drifts, right)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("GLE harmonic autocorrelation solve failed")
    return result


def fit_optimal_gle(frequency_range_au, *, naux=2, grid_size=32, max_nfev=600):
    """Fit canonical GLE potential-energy sampling efficiency on a log grid.

    The objective is log(omega*tau_V) with tau_V the exact harmonic integrated
    potential-energy autocorrelation time.  Its ideal target is zero.  Residuals
    quantify sampling efficiency, not a departure from the canonical measure.
    The scalar analytic limit is supplied by the public parameter module.
    """
    band, naux, grid_size, max_nfev = _settings(
        frequency_range_au, naux, grid_size, max_nfev)
    scale = math.sqrt(float(band[0])) * math.sqrt(float(band[1]))
    normalized = band / scale
    grid = np.geomspace(*normalized, grid_size)
    heldout = np.geomspace(*normalized, 8 * grid_size + 1)
    size = naux + 1
    rates = np.r_[normalized[0], np.geomspace(.5, 5., naux)]
    couplings = np.geomspace(.5, 5., naux)
    initial = np.log(np.r_[rates, couplings])
    lower = np.full(len(initial), math.log(float(normalized[0]) * 1e-4))
    upper = np.full(len(initial), math.log(float(normalized[1]) * 1e4))

    def residual(parameters):
        drift, _ = _arrowhead(parameters, size, thermal=False)
        return np.log(grid * _potential_tau(drift, grid))

    fit = least_squares(residual, initial, bounds=(lower, upper),
                        max_nfev=max_nfev, **_SOLVER_TOLERANCES)
    drift, diffusion = _arrowhead(fit.x, size, thermal=False)
    tau = _potential_tau(drift, heldout, independent=True)
    objective = np.log(heldout * tau)
    rms = float(np.sqrt(np.mean(objective**2)))
    if not fit.success:
        raise ValueError(f"optimal GLE fit did not converge: rms_log_objective={rms:.9g}; increase max_nfev explicitly")
    _, lyapunov_residual = _validate_covariance(drift, diffusion)
    physical_drift = drift * scale
    if not np.isfinite(physical_drift).all() or np.min(np.diag(physical_drift)) <= 0:
        raise ValueError("generated drift cannot be represented in atomic units")
    metadata = {
        "schema": _SCHEMA, "generator": "optimal-sampling-fit", "implementation": _VERSION,
        "frequency_range_au": band.tolist(), "frequency_grid_au": (grid * scale).tolist(),
        "validation_frequency_grid_au": (heldout * scale).tolist(),
        "frequency_convention": "angular", "naux": naux, "grid_size": grid_size,
        "objective": "least squares of log(omega*tau_V); ideal target zero",
        "target": "potential-energy integrated autocorrelation efficiency kappa=1/(omega*tau_V)",
        "parameterization": "symmetric arrowhead A with positive Schur complement; C=I; D=2*A",
        "covariance_normalization": "dimensionless: divide mass-scaled covariance by P/beta",
        "training_log_objective": residual(fit.x).tolist(),
        "validation_log_objective": objective.tolist(),
        "validation_tau_V_au": (tau / scale).tolist(),
        "minimum_efficiency": float(np.min(1 / (heldout * tau))),
        "rms_log_objective": rms, "lyapunov_relative_residual": lyapunov_residual,
        "fit_converged": True, "physical_converged": False,
        "scope": "canonical harmonic sampling efficiency; local fit, not a certified global optimum",
        "solver": _solver_metadata(fit, initial, lower, upper, max_nfev),
    }
    return {
        "A": physical_drift.tolist(), "C": np.eye(size).tolist(),
        "provenance": f"PyRPMD generated {metadata['generator']} {_VERSION}; no third-party matrix data",
        "parameter_metadata": metadata,
    }


__all__ = ["fit_piglet", "fit_optimal_gle"]
