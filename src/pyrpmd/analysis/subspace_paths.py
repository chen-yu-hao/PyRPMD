"""Physical path cores for the subspace correction that do not use centroid geometries.

The historical core of the reaction path is the mean ring-polymer centroid of
the orientation-locked umbrella windows.  A centroid is a thermal average,
not a geometry: a librating or ring-opening coordinate has a centroid that is
shorter than any bond the polymer visits, and near the reactants the mean of
freely rotating fragments is not a molecule at all.  The cores here are
stationary-point constructions on the potential itself:

``mep``
    the intrinsic reaction coordinate.  The transition-state geometry of the
    rate run is refined to a first-order saddle (partitioned rational
    function optimization with a finite-difference Hessian, rigid-body modes
    removed), and the core is the saddle displaced by ``+-displacement``
    along its imaginary mode.  :func:`assemble_reaction_path` then follows
    steepest descent from both sides: a minimum-energy path everywhere,
    which is what the vibrationally adiabatic normal-cut model assumes.
``pmf_minimum``
    for every umbrella window, the minimum of ``V`` on that window's
    ``xi = const`` surface, marched outwards from the window at the
    transition state so that neighbouring points stay in one basin.  It
    covers the full sampled ``xi`` range of the PMF with physical bond
    lengths.  It is a distinguished-coordinate path: on it ``grad V`` is
    parallel to ``grad xi``, not in general to the path tangent.

All work is in mass-weighted Cartesian coordinates; rigid-body modes are
removed with :func:`pyrpmd.analysis.centroid_path.rigid_body_modes`.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from .centroid_path import rigid_body_modes


def _vibrational_basis(x, masses):
    """Orthonormal mass-weighted basis of the vibrational space at ``x``."""
    from scipy.linalg import null_space

    rigid = rigid_body_modes(np.asarray(x).reshape(-1, 3), masses)
    return null_space(rigid.T)


def mass_weighted_hessian(gradient: Callable[[np.ndarray], np.ndarray], x: Any, masses: Any, *,
                          displacement: float = 5e-3) -> np.ndarray:
    """Central-difference mass-weighted Hessian ``[3N, 3N]`` at ``x``.

    ``displacement`` is the Cartesian displacement (bohr) of the lightest
    atom; the mass-weighted step is the same for every coordinate.
    """
    x = np.asarray(x, dtype=float).ravel()
    sqrt_m = np.sqrt(np.repeat(np.asarray(masses, dtype=float), 3))
    h = float(displacement) * float(np.sqrt(np.min(masses)))
    columns = []
    for i in range(x.size):
        e = np.zeros(x.size)
        e[i] = h / sqrt_m[i]
        plus = np.asarray(gradient(x + e), dtype=float).ravel() / sqrt_m
        minus = np.asarray(gradient(x - e), dtype=float).ravel() / sqrt_m
        columns.append((plus - minus) / (2.0 * h))
    hessian = np.column_stack(columns)
    return 0.5 * (hessian + hessian.T)


def refine_saddle(gradient: Callable[[np.ndarray], np.ndarray], x0: Any, masses: Any, *,
                  trust_radius: float = 5.0, gradient_tolerance: float = 1e-5, max_iterations: int = 100,
                  displacement: float = 5e-3) -> tuple[np.ndarray, dict]:
    """Refine ``x0`` to a first-order saddle point by partitioned RFO.

    Steps are taken in the vibrational space: uphill along the lowest
    Hessian mode (rational-function shift for a maximum), downhill along all
    others (shift for a minimum), scaled to at most ``trust_radius``
    (mass-weighted).  Converged when the largest Cartesian component of the
    vibrational gradient is below ``gradient_tolerance`` (hartree/bohr).
    Returns the saddle and a record with the iterations, the gradient, the
    imaginary frequency (hartree) and the number of negative curvatures.
    ``saddle`` in the record is true only for exactly one negative mode.
    """
    x = np.asarray(x0, dtype=float).ravel().copy()
    masses = np.asarray(masses, dtype=float)
    sqrt_m = np.sqrt(np.repeat(masses, 3))
    record = {"iterations": 0, "converged": False}
    for iteration in range(int(max_iterations) + 1):
        basis = _vibrational_basis(x, masses)
        g = basis.T @ (np.asarray(gradient(x), dtype=float).ravel() / sqrt_m)
        cartesian = np.abs((basis @ g) * sqrt_m).max() if g.size else 0.0
        hessian = basis.T @ mass_weighted_hessian(gradient, x, masses, displacement=displacement) @ basis
        values, vectors = np.linalg.eigh(hessian)
        record.update(iterations=iteration, gradient_max=float(cartesian))
        if cartesian < gradient_tolerance:
            record["converged"] = True
            break
        if iteration == max_iterations:
            break
        q = vectors.T @ g
        # Maximize along the lowest mode ...
        lam0, g0 = values[0], q[0]
        mu_max = 0.5 * lam0 + 0.5 * np.sqrt(lam0 ** 2 + 4.0 * g0 ** 2)
        step = np.zeros_like(q)
        step[0] = -g0 / (lam0 - mu_max) if lam0 != mu_max else 0.0
        # ... and minimize along the others (lowest root of the augmented Hessian).
        if len(values) > 1:
            augmented = np.zeros((len(values), len(values)))
            augmented[:-1, :-1] = np.diag(values[1:])
            augmented[:-1, -1] = augmented[-1, :-1] = q[1:]
            mu_min = float(np.linalg.eigvalsh(augmented)[0])
            step[1:] = -q[1:] / (values[1:] - mu_min)
        length = float(np.linalg.norm(step))
        if length > trust_radius:
            step *= trust_radius / length
        x = x + (basis @ (vectors @ step)) / sqrt_m
    negative = int(np.count_nonzero(values < 0))
    record.update(negative_modes=negative, saddle=bool(record["converged"] and negative == 1),
                  imaginary_frequency=float(np.sqrt(-values[0])) if values[0] < 0 else None,
                  lowest_curvature=float(values[0]))
    record["mode"] = (basis @ vectors[:, 0]).tolist()
    return x, record


def mep_core(run: Any, options: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    """Saddle point displaced along its imaginary mode, ordered reactants -> products.

    ``run`` provides ``gradient``, ``xi_value_gradient``, ``system.geometry``
    (the transition-state reference of the rate coordinate) and
    ``system.masses``.  Options: ``mep_displacement`` (mass-weighted,
    default a tenth of ``step``), ``ts_gradient_tolerance`` (hartree/bohr,
    default 1e-5) and ``hessian_displacement``.
    """
    masses = np.asarray(run.system.masses, dtype=float)
    sqrt_m = np.sqrt(np.repeat(masses, 3))
    step = float(options.get("step", 10.0))
    displacement = float(options.get("mep_displacement") or 0.1 * step)
    saddle, record = refine_saddle(run.gradient, np.asarray(run.system.geometry, dtype=float).ravel(), masses,
                                   trust_radius=0.5 * step,
                                   gradient_tolerance=float(options.get("ts_gradient_tolerance", 1e-5)),
                                   displacement=float(options.get("hessian_displacement", 5e-3)))
    if not record["saddle"]:
        raise ValueError(
            "path.source = 'mep' needs a first-order saddle near the transition-state geometry of the rate run; "
            f"refinement ended with {record['negative_modes']} negative curvature(s) and largest gradient "
            f"{record['gradient_max']:.3g} Eh/bohr after {record['iterations']} iterations")
    mode = np.asarray(record.pop("mode"))
    ends = [saddle - displacement * mode / sqrt_m, saddle + displacement * mode / sqrt_m]
    values = [run.xi_value_gradient(end)[0] for end in ends]
    if values[0] > values[1]:
        ends = ends[::-1]
    core = np.vstack([ends[0], saddle, ends[1]])
    moved = float(np.linalg.norm((saddle - np.asarray(run.system.geometry, dtype=float).ravel()) * sqrt_m))
    record.update(displacement=displacement, saddle_shift_mass_weighted=moved,
                  saddle_energy=float(run.energy(saddle)), saddle_xi=float(run.xi_value_gradient(saddle)[0]),
                  xi_of_ends=sorted(float(v) for v in values))
    return core, record


def pmf_minimum_core(run: Any, xi_values: Any, options: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    """Minimum of ``V`` on each window's ``xi`` surface, marched out from the transition state.

    ``xi_values`` are the umbrella window centres.  The window nearest to
    ``xi`` of the transition-state geometry is minimized first, starting
    there; every further window starts from its relaxed neighbour, so the
    path stays in one basin and is connected.  Returns the points ordered
    by increasing ``xi`` and a record of the constraint residuals.
    """
    from scipy.optimize import minimize

    grid = np.unique(np.asarray(xi_values, dtype=float))
    if len(grid) < 2:
        raise ValueError("path.source = 'pmf_minimum' needs at least two umbrella windows")
    masses = np.asarray(run.system.masses, dtype=float)
    sqrt_m = np.sqrt(np.repeat(masses, 3))
    seed = np.asarray(run.system.geometry, dtype=float).ravel()
    start = int(np.argmin(np.abs(grid - run.xi_value_gradient(seed)[0])))
    tolerance = float(options.get("pmf_constraint_tolerance", 1e-8))

    def relax(target, x0):
        # Mass-weighted coordinates keep the conditioning of light and heavy atoms alike.
        fun = lambda q: (float(run.energy(q / sqrt_m)), np.asarray(run.gradient(q / sqrt_m), float).ravel() / sqrt_m)
        constraint = {"type": "eq",
                      "fun": lambda q: run.xi_value_gradient(q / sqrt_m)[0] - target,
                      "jac": lambda q: np.asarray(run.xi_value_gradient(q / sqrt_m)[1], float).ravel() / sqrt_m}
        result = minimize(fun, x0 * sqrt_m, jac=True, method="SLSQP", constraints=[constraint],
                          options={"ftol": 1e-14, "maxiter": 500})
        x = result.x / sqrt_m
        return x, abs(run.xi_value_gradient(x)[0] - target), bool(result.success)

    points, residuals, failures = {}, {}, []
    for order in (range(start, -1, -1), range(start + 1, len(grid))):
        previous = seed if start in order else points[start]
        for k in order:
            x, residual, success = relax(grid[k], previous)
            if not success or residual > tolerance:
                failures.append(float(grid[k]))
            points[k], residuals[k], previous = x, residual, x
    core = np.vstack([points[k] for k in range(len(grid))])
    energies = [float(run.energy(p)) for p in core]
    return core, {"windows": int(len(grid)), "xi": grid.tolist(), "start_xi": float(grid[start]),
                  "max_constraint_residual": float(max(residuals.values())), "failed_xi": failures,
                  "energies": energies,
                  "definition": "argmin V on xi = xi_window, continued from the neighbouring window"}
