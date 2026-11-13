"""Exact quantum thermal rate constants for 1-3D subspace Hamiltonians.

The rate follows Miller, Schwartz and Tromp with an absorbing potential:

    k Q_r = integral_0^infty Re Tr[F_beta exp(i H^dag t) F exp(-i H t)] dt,

``F = i[H_0, h]`` the flux through ``s_0 = dividing_surface`` (``h`` the
product-side projector), ``F_beta = exp(-beta H_0 / 2) F exp(-beta H_0 / 2)``
and ``H = H_0 - i eps`` with an absorbing potential ``eps`` at both ends of the
reaction coordinate.  ``H_0`` uses a Colbert--Miller sinc DVR on every axis.

Two exact routes are implemented:

``"sylvester"``
    Dense.  ``X = int_0^infty exp(i H^dag t) F exp(-i H t) dt`` solves
    ``H^dag X - X H = i F``; one complex Schur form and a triangular
    Sylvester solve give ``Re Tr[F_beta X]`` with no time truncation, so
    long-lived resonances are treated exactly.  Cost ``O(N^3)`` for ``N``
    grid points (default limit 1600).
``"flux_eigenstates"``
    Iterative, for larger 2D/3D grids.  The few significant eigenpairs
    ``(f_k, v_k)`` of ``F_beta`` are found by Lanczos with ``exp(-beta H_0/2)``
    applied by a Chebyshev expansion; each ``v_k`` is propagated with a
    second-order split operator under ``H``.  ``int <F> dt`` per state equals
    the norm absorbed on the product side minus its initial product
    population.  Norm still trapped at ``max_time`` is assigned by the
    product/reactant absorption branching ratio of the second half of the
    run, and the estimates at ``T/2``, ``3T/4`` and ``T`` are reported as a
    convergence check.

Normalization (identical to the ring-polymer solver):

``reactant='asymptotic'``
    ``Q_r = sqrt(m_0/(2 pi beta)) Q_perp``, the per-length translational
    density times the exact transverse partition function at
    ``reactant_reference``; ``k`` then has units of axis-0 length per time.
``reactant='well'``
    ``Q_r = Tr[exp(-beta H_0)(1 - h)]`` on the grid.

Grid spacing, box, absorbing potential and (iterative route) time step remain
numerical parameters that a caller must converge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
from time import perf_counter

import numpy as np
from scipy.linalg import eigh, schur
from scipy.linalg.lapack import ztrsyl
from scipy.special import ive

from .subspace_model import SubspaceModel


def sinc_dvr_kinetic(axis: np.ndarray, mass: float) -> np.ndarray:
    """Colbert--Miller sinc-DVR kinetic matrix on a uniform grid."""
    axis = np.asarray(axis, dtype=float)
    n = axis.size
    dx = float(axis[1] - axis[0])
    index = np.arange(n)
    difference = index[:, None] - index[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = 2.0 * np.where(difference % 2 == 0, 1.0, -1.0) / difference.astype(float) ** 2
    matrix[np.diag_indices(n)] = np.pi**2 / 3.0
    return matrix / (2.0 * float(mass) * dx * dx)


def absorbing_potential(axis0: np.ndarray, *, width: Any, strength: Any, power: int = 2) -> np.ndarray:
    """Power-law absorbing potential over ``width`` at both ends of axis 0.

    ``width`` and ``strength`` are one value for both ends or a pair
    ``(low end, high end)``.
    """
    axis0 = np.asarray(axis0, dtype=float)
    (w_lo, w_hi), (s_lo, s_hi) = _pair(width), _pair(strength)
    lower, upper = axis0[0] + w_lo, axis0[-1] - w_hi
    eps = np.zeros_like(axis0)
    left, right = axis0 < lower, axis0 > upper
    eps[left] = s_lo * ((lower - axis0[left]) / w_lo) ** power
    eps[right] = s_hi * ((axis0[right] - upper) / w_hi) ** power
    return eps


def _pair(value) -> tuple[float, float]:
    """``value`` for both ends, or an explicit ``(low end, high end)`` pair."""
    if np.ndim(value) == 0:
        return float(value), float(value)
    values = tuple(float(v) for v in np.asarray(value, dtype=float).ravel())
    if len(values) != 2:
        raise ValueError("an absorber width or strength is one value or a (low end, high end) pair")
    return values


@dataclass(frozen=True)
class QuantumRateResult:
    """Exact thermal rate of a subspace Hamiltonian.

    ``rate = flux / reactant_partition``; both carry the same energy shift
    ``exp(beta * energy_shift)``.  ``convergence`` lists flux estimates that
    must agree for the result to be trusted (iterative route only).
    """

    beta: float
    rate: float
    flux: float
    reactant_partition: float
    method: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def to_dict(self) -> dict:
        return {"beta": self.beta, "rate": self.rate, "flux": self.flux,
                "reactant_partition": self.reactant_partition, "method": self.method,
                "diagnostics": _json_safe(self.diagnostics)}


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class _GridOperators:
    """Kronecker-sum sinc-DVR Hamiltonian applied to batches ``[k, *shape]``."""

    def __init__(self, model: SubspaceModel):
        self.model = model
        self.shape = model.shape
        self.size = int(np.prod(self.shape))
        self.kinetic = [sinc_dvr_kinetic(a, m) for a, m in zip(model.axes, model.masses)]
        self.potential = np.asarray(model.values, dtype=float)
        self.product = np.broadcast_to(
            (model.axes[0] > model.dividing_surface).astype(float).reshape((-1,) + (1,) * (model.dimension - 1)),
            self.shape,
        ).copy()

    def apply_axis(self, matrix: np.ndarray, psi: np.ndarray, axis: int) -> np.ndarray:
        return np.moveaxis(np.tensordot(matrix, psi, axes=([1], [axis + 1])), 0, axis + 1)

    def kinetic_apply(self, psi: np.ndarray) -> np.ndarray:
        out = np.zeros_like(psi)
        for axis, matrix in enumerate(self.kinetic):
            out += self.apply_axis(matrix, psi, axis)
        return out

    def flux_apply(self, psi: np.ndarray) -> np.ndarray:
        # Only the axis-0 kinetic energy fails to commute with h(s_0).
        t0 = self.kinetic[0]
        return 1j * (self.apply_axis(t0, self.product * psi, 0) - self.product * self.apply_axis(t0, psi, 0))

    def dense_hamiltonian(self, keep=None) -> np.ndarray:
        """Dense ``H`` on all grid points, or on the flat indices ``keep``.

        Dropping points is the energy-truncated DVR: excluded points act as
        infinite walls, exact once the truncation energy is high enough.
        """
        if keep is None:
            h = np.zeros((self.size, self.size), order="F")
            h[np.diag_indices(self.size)] = self.potential.ravel()
            for axis, matrix in enumerate(self.kinetic):
                right = int(np.prod(self.shape[axis + 1:]))
                row = np.arange(self.shape[axis]) * right
                for start in range(0, self.size, len(row) * right):
                    for offset in range(right):
                        index = row + start + offset
                        h[np.ix_(index, index)] += matrix
            return h
        index = np.unravel_index(np.asarray(keep), self.shape)
        h = np.zeros((len(keep), len(keep)), order="F")
        h[np.diag_indices(len(keep))] = self.potential.ravel()[keep]
        for axis, matrix in enumerate(self.kinetic):
            # Couple only points on the same grid line. Avoid N x N masks,
            # gathered kinetic matrices and full Kronecker temporaries.
            labels = np.zeros(len(keep), dtype=np.int64)
            for other in range(len(self.shape)):
                if other != axis:
                    labels = labels * self.shape[other] + index[other]
            order = np.argsort(labels, kind="stable")
            cuts = np.flatnonzero(np.diff(labels[order])) + 1
            for group in np.split(order, cuts):
                h[np.ix_(group, group)] += matrix[np.ix_(index[axis][group], index[axis][group])]
        return h


def _transverse_partition(model: SubspaceModel, beta: float, shift: float) -> tuple[float, dict]:
    reference = model.reactant_reference
    if model.dimension == 1:
        energy = float(model.energy(np.array([[reference]]))[0])
        return float(np.exp(-beta * (energy - shift))), {"reference_energy": energy}
    axes = model.axes[1:]
    if np.prod([len(a) for a in axes]) > 5000:
        raise ValueError("transverse reactant slice exceeds 5000 points; use a coarser transverse grid")
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    points = np.concatenate([np.full(mesh.shape[:-1] + (1,), reference), mesh], axis=-1)
    slice_potential = np.asarray(model.energy(points), dtype=float).ravel()
    hamiltonian = np.diag(slice_potential)
    shape = tuple(len(a) for a in axes)
    for index, (axis, mass) in enumerate(zip(axes, model.masses[1:])):
        left = int(np.prod(shape[:index]))
        right = int(np.prod(shape[index + 1:]))
        hamiltonian = hamiltonian + np.kron(np.kron(np.eye(left), sinc_dvr_kinetic(axis, mass)), np.eye(right))
    levels = np.linalg.eigvalsh(hamiltonian)
    weights = np.exp(-beta * (levels - shift))
    return float(weights.sum()), {
        "reference_energy_levels": levels[:8].tolist(),
        "transverse_tail_weight": float(weights[-1] / weights.sum()),
    }


def _reactant_partition(model, beta, shift, *, energies=None, vectors=None, operators=None, tolerance=1e-12,
                        product=None):
    if model.reactant == "asymptotic":
        transverse, info = _transverse_partition(model, beta, shift)
        value = np.sqrt(model.masses[0] / (2.0 * np.pi * beta)) * transverse
        info.update(normalization="per unit length of axis 0 at reactant_reference")
        return float(value), info
    reactant = 1.0 - (operators.product.ravel() if product is None else product)
    if energies is None:
        energies, vectors = _lowest_states(operators, beta, shift, tolerance)
    weights = np.exp(-beta * (energies - shift))
    populations = np.sum(vectors * vectors * reactant[:, None], axis=0)
    return float(np.sum(weights * populations)), {
        "normalization": "Tr[exp(-beta H0)(1-h)] on the grid",
        "boltzmann_tail_weight": float(weights[-1] / weights.sum()),
        "states": int(len(energies)),
    }


def _lowest_states(operators, beta, shift, tolerance):
    from scipy.sparse.linalg import LinearOperator, eigsh

    n = operators.size
    potential = operators.potential

    def matvec(v):
        psi = v.reshape((1,) + operators.shape)
        return (operators.kinetic_apply(psi) + potential * psi).ravel()

    op = LinearOperator((n, n), matvec=matvec, dtype=float)
    count = min(32, n - 2)
    while True:
        energies, vectors = eigsh(op, k=count, which="SA", tol=1e-10)
        order = np.argsort(energies)
        energies, vectors = energies[order], vectors[:, order]
        if np.exp(-beta * (energies[-1] - energies[0])) < tolerance or count >= min(2000, n - 2):
            return energies, vectors
        count = min(2 * count, n - 2)


def _slice_density(model, operators, keep, energies, vectors, beta, shift):
    """Quantum line density on the axis-0 grid line nearest ``reactant_reference``.

    ``int <x|exp(-beta H)|x> dl`` along the line: the normalization shared with
    the RPMD constrained-centroid TI along axis 0.
    """
    axis0 = model.axes[0]
    row = int(np.argmin(abs(axis0 - model.reactant_reference)))
    index = np.unravel_index(keep, operators.shape)
    on_line = index[0] == row
    cell = float(np.prod(model.spacing))
    weights = np.exp(-beta * (energies - shift))
    density = np.sum(vectors[on_line] ** 2 * weights[None, :], axis=1) / cell
    line = float(np.sum(density) * cell / model.spacing[0])
    return line, {"normalization": "quantum line density on the reference slice",
                  "reference_line": float(axis0[row]), "slice_points": int(on_line.sum())}


def _triangular_sylvester(triangular, rhs, block_size=64):
    """Solve T^H X - X T = rhs with LAPACK tiles and BLAS matrix updates.

    The tile traversal follows the lower/upper triangular dependencies.
    Rare LAPACK scaling or perturbed eigenvalues use the original full solve,
    retaining LAPACK's global overflow/near-degeneracy handling.
    """
    n = len(rhs)
    if n <= block_size:
        return (*ztrsyl(triangular, triangular, rhs, trana="C", tranb="N", isgn=-1), "lapack")
    solution = np.empty((n, n), dtype=complex, order="F")
    for i in range(0, n, block_size):
        rows = slice(i, min(i + block_size, n))
        for j in range(0, n, block_size):
            cols = slice(j, min(j + block_size, n))
            tile = np.array(rhs[rows, cols], order="F", copy=True)
            if i:
                tile -= triangular[:i, rows].conj().T @ solution[:i, cols]
            if j:
                tile += solution[rows, :j] @ triangular[:j, cols]
            value, scale, info = ztrsyl(triangular[rows, rows], triangular[cols, cols], tile,
                                       trana="C", tranb="N", isgn=-1, overwrite_c=True)
            if scale != 1.0 or info != 0 or not np.isfinite(value).all():
                return (*ztrsyl(triangular, triangular, rhs, trana="C", tranb="N", isgn=-1),
                        "lapack_fallback")
            solution[rows, cols] = value
    return solution, 1.0, 0, "blocked_lapack"


def _dense_memory_mb(model, retained, normalization="default"):
    transverse = int(np.prod(model.shape[1:])) if model.reactant == "asymptotic" and normalization == "default" else 0
    return 64 + (192 * retained**2 + 64 * np.prod(model.shape) + 32 * transverse**2) / 2**20


def _iterative_memory_mb(model, max_flux_states=256):
    size = int(np.prod(model.shape))
    states = min(size, int(max_flux_states))
    return 64 + (256 * size * states + 64 * sum(n**2 for n in model.shape)) / 2**20


# Options that only the dense Sylvester route implements.
_DENSE_ONLY_OPTIONS = ("truncation_energy", "energy_floor", "slice normalization")


class DenseRouteUnavailable(ValueError):
    """The requested options need the dense route, but the grid does not fit it.

    A ``ValueError`` for compatibility.  ``diagnostic`` is the solver plan of
    :func:`quantum_solver_plan`: retained points, ``dense_limit``, the dense
    memory estimate, ``required_dense_limit`` and the supported alternatives.
    """

    def __init__(self, plan):
        self.diagnostic = dict(plan)
        super().__init__(plan["reason"])


def _route_plan(model, retained, *, method, dense_limit, dense_only, normalization, max_flux_states,
                memory_limit_mb):
    if method not in ("auto", "sylvester", "flux_eigenstates"):
        raise ValueError("method must be 'auto', 'sylvester' or 'flux_eigenstates'")
    if isinstance(dense_limit, bool) or not isinstance(dense_limit, (int, float, np.integer, np.floating)) \
            or not np.isfinite(dense_limit) or dense_limit < 1:
        raise ValueError("dense_limit must be a positive number of grid points")
    dense_limit = int(dense_limit)
    route = method if method != "auto" else ("sylvester" if retained <= dense_limit else "flux_eigenstates")
    dense_mb = float(_dense_memory_mb(model, retained, normalization))
    route_mb = dense_mb if route == "sylvester" else float(_iterative_memory_mb(model, max_flux_states))
    plan = {
        "grid_shape": [int(n) for n in model.shape], "grid_points": int(np.prod(model.shape)),
        "retained_points": int(retained), "method": method, "route": route, "dense_limit": dense_limit,
        "dense_only_options": list(dense_only), "dense_memory_mb": dense_mb, "estimated_memory_mb": route_mb,
        "memory_limit_mb": None if memory_limit_mb is None else float(memory_limit_mb),
        "required_dense_limit": int(retained) if dense_only else None,
        "feasible": True, "status": "ok", "reason": None, "alternatives": [],
    }
    dense_gib = dense_mb / 1024.0
    reduce = ("reduce the retained grid (a smaller box, a coarser grid or a lower truncation energy) "
              "and check that the rate is converged with respect to it")
    if route == "flux_eigenstates" and dense_only:
        plan.update(
            feasible=False, status="dense_limit_exceeded" if method == "auto" else "dense_route_required",
            reason=(f"{', '.join(dense_only)} need the dense route, but {retained} grid points remain "
                    f"after truncation and dense_limit is {dense_limit} (the dense route needs about "
                    f"{dense_gib:.1f} GiB)"),
            alternatives=[f"raise dense_limit to at least {retained} if about {dense_gib:.1f} GiB is available",
                          reduce])
    elif route == "sylvester" and retained > max(dense_limit, 6000):
        plan.update(feasible=False, status="dense_limit_exceeded", required_dense_limit=int(retained),
                    reason=(f"grid too large for the dense Sylvester route: {retained} retained points "
                            f"exceed dense_limit={dense_limit} (about {dense_gib:.1f} GiB)"),
                    alternatives=[f"raise dense_limit to at least {retained} if about {dense_gib:.1f} GiB "
                                  "is available", reduce])
    elif memory_limit_mb is not None and route_mb > float(memory_limit_mb):
        plan.update(feasible=False, status="memory_limit_exceeded",
                    reason=(f"the {route} route needs about {route_mb:.1f} MiB for {retained} retained points; "
                            f"the memory budget is {float(memory_limit_mb):.1f} MiB"),
                    alternatives=["raise the memory budget", reduce])
    return plan


def quantum_solver_plan(model: SubspaceModel, *, method: str = "auto", dense_limit: int = 1600,
                        truncation_energy: Any = None, energy_floor: float | None = None,
                        normalization: str = "default", max_flux_states: int = 256,
                        memory_limit_mb: float | None = None, **_solver_options) -> dict:
    """Route, size and memory of :func:`exact_quantum_rate` for ``model``, without solving it.

    Takes the same keywords (other solver options are ignored) and costs one
    comparison per grid point.  ``feasible`` is false, with ``status``
    (``dense_limit_exceeded``, ``dense_route_required`` or
    ``memory_limit_exceeded``), ``reason`` and ``alternatives``, exactly when
    the solve would be refused for its route or size; ``memory_limit_mb``
    (optional) also bounds the iterative route, as the task scheduler does.
    """
    if not isinstance(model, SubspaceModel):
        raise TypeError("model must be a SubspaceModel")
    if normalization not in ("default", "slice"):
        raise ValueError("normalization must be 'default' or 'slice'")
    retained = int(np.prod(model.shape))
    if truncation_energy is not None:
        threshold = _truncation_threshold(model, truncation_energy)
        retained = int(np.count_nonzero(np.asarray(model.values, dtype=float) <= threshold))
    used = (truncation_energy is not None, energy_floor is not None, normalization == "slice")
    return _route_plan(model, retained, method=method, dense_limit=dense_limit,
                       dense_only=[name for name, on in zip(_DENSE_ONLY_OPTIONS, used) if on],
                       normalization=normalization, max_flux_states=max_flux_states,
                       memory_limit_mb=memory_limit_mb)


def _dense_rate_matrix(
    hamiltonian,
    eps,
    product,
    beta,
    shift=0.0,
    *,
    partition=None,
    reactant_weights=None,
    energy_floor=None,
):
    """Evaluate the dense Sylvester flux for a general matrix Hamiltonian.

    ``hamiltonian`` is the finite-dimensional Hermitian ``H_0`` matrix,
    ``product`` is the diagonal of the product-side projector, and ``eps``
    is the diagonal absorbing potential.  The matrix need not come from a
    tensor-product DVR; this is the interface used by the adiabatic-channel
    solver.  ``partition`` may be a number or a callable receiving
    ``(energies, eigenvectors)``.  If it is omitted, ``reactant_weights``
    supplies the diagonal reactant population factors, which is the exact
    finite-grid normalization used by the original route.
    """
    hamiltonian = np.asarray(hamiltonian)
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must be a square matrix")
    if not np.isfinite(hamiltonian).all() or not np.allclose(
        hamiltonian, hamiltonian.conj().T, atol=1e-11, rtol=1e-10
    ):
        raise ValueError("hamiltonian must be finite and Hermitian")
    n = hamiltonian.shape[0]
    product = np.asarray(product, dtype=float).reshape(-1)
    eps = np.asarray(eps, dtype=float).reshape(-1)
    if product.size != n or eps.size != n:
        raise ValueError("product and eps must have one value per Hamiltonian state")
    if not np.isfinite(product).all() or not np.isfinite(eps).all() or np.any(eps < 0):
        raise ValueError("product and eps must be finite; eps must be nonnegative")
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive")
    shift = float(shift)
    started = perf_counter()
    energies, vectors = eigh(hamiltonian)
    diagonalized = perf_counter()
    half = np.exp(-0.5 * beta * (energies - shift))
    if energy_floor is not None:
        energy_floor = float(energy_floor)
        if not np.isfinite(energy_floor):
            raise ValueError("energy_floor must be finite")
        half = np.where(energies >= energy_floor, half, 0.0)
    # The partition needs the eigenvectors; evaluate it first so they can be
    # released before the Schur and Sylvester steps allocate their matrices.
    if partition is None:
        if reactant_weights is None:
            raise ValueError("partition or reactant_weights is required")
        weights = np.asarray(reactant_weights, dtype=float).reshape(-1)
        if weights.size != n or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("reactant_weights must be finite and nonnegative")
        populations = np.sum(vectors.conj() * vectors * weights[:, None], axis=0).real
        value = float(np.sum(np.exp(-beta * (energies - shift)) * populations))
        partition_info = {"normalization": "reactant weights in matrix basis", "states": int(n)}
    elif callable(partition):
        value = partition(energies, vectors)
        if isinstance(value, tuple):
            value, partition_info = value
        else:
            partition_info = {"normalization": "caller supplied matrix partition"}
        value = float(value)
    else:
        value = float(partition)
        partition_info = {"normalization": "caller supplied matrix partition"}
    if not np.isfinite(value) or value <= 0:
        raise FloatingPointError("matrix partition must be finite and positive")
    lowest_levels = energies[:6].tolist()
    states_below_floor = int(np.sum(energies < energy_floor)) if energy_floor is not None else 0
    # K = [H, h] is real antisymmetric for the real Hamiltonians used by the
    # DVR and channel constructions.  The formula below also accepts a
    # complex Hermitian matrix and uses the Hermitian commutator convention.
    # Each matrix is released once used, so the peak stays near four complex
    # N x N arrays (the Schur pair, the right-hand side and the solution).
    commutator = hamiltonian * product[None, :]
    commutator -= product[:, None] * hamiltonian
    boltzmann_half = (vectors * half[None, :]) @ vectors.conj().T
    del vectors, energies
    thermal_real = boltzmann_half @ commutator @ boltzmann_half
    del boltzmann_half
    absorbing = np.array(hamiltonian, dtype=complex, order="F")
    absorbing[np.diag_indices_from(absorbing)] -= 1j * eps
    schur_started = perf_counter()
    triangular, unitary = schur(absorbing, output="complex", overwrite_a=True)
    schur_finished = perf_counter()
    del absorbing
    # Right-hand side i U^+ F U = -U^+ K U, with F=iK.
    rhs = unitary.conj().T @ commutator
    del commutator
    rhs = rhs @ unitary
    rhs *= -1.0
    sylvester_started = perf_counter()
    solution, scale, info, solver = _triangular_sylvester(triangular, rhs)
    sylvester_finished = perf_counter()
    del rhs, triangular
    if info < 0:
        raise RuntimeError(f"ztrsyl failed with info={info}")
    solution /= scale
    projected = unitary.conj().T @ thermal_real
    projected = projected @ unitary
    flux = float(np.real(1j * np.einsum("ij,ji->", projected, solution)))
    del projected, solution, unitary
    # Tr[F_beta h] vanishes for an exact thermal flux operator; a nonzero value
    # measures how well the grid represents the dividing surface.
    side_check = float(np.real(1j * np.sum(np.diag(thermal_real) * product)))
    del thermal_real
    return flux, value, {
        "grid_points": int(n),
        "retained_points": int(n),
        "sylvester_scale": float(scale),
        "sylvester_info": int(info),
        "sylvester_solver": solver,
        "timing_seconds": {"assembly_and_eigh": diagonalized - started,
                           "schur": schur_finished - schur_started,
                           "sylvester": sylvester_finished - sylvester_started,
                           "dense_total": perf_counter() - started},
        "flux_side_t0": side_check,
        "lowest_levels": lowest_levels,
        "energy_floor": energy_floor,
        "states_below_floor": states_below_floor,
        "partition": partition_info,
    }


def _dense_rate(model, operators, eps, beta, shift, *, keep=None, energy_floor=None, normalization="default"):
    keep = np.arange(operators.size) if keep is None else np.asarray(keep)
    hamiltonian = operators.dense_hamiltonian(keep if len(keep) < operators.size else None)
    product = operators.product.ravel()[keep]
    eps_keep = np.broadcast_to(eps, operators.shape).ravel()[keep]
    if normalization == "slice":
        flux, partition, info = _dense_rate_matrix(
            hamiltonian, eps_keep, product, beta, shift,
            partition=lambda e, v: _slice_density(model, operators, keep, e, v, beta, shift),
            energy_floor=energy_floor,
        )
        info_q = info["partition"]
    else:
        def matrix_partition(energies, vectors):
            full_vectors = vectors
            if len(keep) < operators.size:
                full_vectors = np.zeros((operators.size, vectors.shape[1]))
                full_vectors[keep] = vectors
            return _reactant_partition(model, beta, shift, energies=energies, vectors=full_vectors,
                                       operators=operators)
        flux, partition, info = _dense_rate_matrix(
            hamiltonian, eps_keep, product, beta, shift,
            partition=matrix_partition, energy_floor=energy_floor,
        )
        info_q = info.pop("partition", {})
    if normalization == "slice":
        info_q = info.pop("partition", info_q)
    info["partition"] = info_q
    info["grid_points"] = operators.size
    info["retained_points"] = int(len(keep))
    return flux, partition, info


class _Chebyshev:
    """``exp(-tau (H_c - E_0)) psi`` with the potential capped at ``cutoff``."""

    def __init__(self, operators, shift, cutoff, tau, tolerance):
        self.ops = operators
        self.capped = np.minimum(operators.potential, cutoff)
        kinetic_max = sum(float(np.linalg.eigvalsh(t)[-1]) for t in operators.kinetic)
        self.emin = float(shift)
        self.emax = float(self.capped.max() + kinetic_max)
        self.half = 0.5 * (self.emax - self.emin)
        self.center = 0.5 * (self.emax + self.emin)
        a = tau * self.half
        orders = np.arange(int(4 * np.sqrt(max(a, 1.0)) * 8 + 64))
        while True:
            values = ive(orders, a)
            keep = np.flatnonzero(values >= tolerance * values[0])
            if keep[-1] < len(orders) - 1:
                break
            orders = np.arange(2 * len(orders))
        count = int(keep[-1]) + 2
        self.coefficients = (2.0 - (orders[:count] == 0)) * np.where(orders[:count] % 2, -1.0, 1.0) * values[:count]
        self.order = count

    def _scaled(self, psi):
        return (self.ops.kinetic_apply(psi) + (self.capped - self.center) * psi) / self.half

    def __call__(self, psi):
        previous = psi
        current = self._scaled(psi)
        result = self.coefficients[0] * previous + self.coefficients[1] * current
        for coefficient in self.coefficients[2:]:
            previous, current = current, 2.0 * self._scaled(current) - previous
            result = result + coefficient * current
        return result


def _flux_eigenstates(operators, boltzmann, count, tolerance, max_states, *, seed=20261107):
    """Significant eigenpairs of ``F_beta = B F B`` by block subspace iteration.

    ``F_beta`` is Hermitian with a rapidly decaying spectrum of ``+-f``
    pairs.  A random block is pushed through ``F_beta`` (two power steps),
    orthonormalized, and the Rayleigh--Ritz pairs are accepted when every
    pair above ``tolerance * max|f|`` has a residual below ``1e-6 max|f|`` and
    the block resolves the spectrum down to that tolerance.  Otherwise the
    block grows.  Each application acts on the whole block at once.
    """
    rng = np.random.default_rng(seed)
    n = operators.size
    shape = operators.shape

    def apply(block):
        return boltzmann(operators.flux_apply(boltzmann(block)))

    def orthonormal(block):
        q, _ = np.linalg.qr(block.reshape(len(block), n).T)
        return q.T.reshape((-1,) + shape)

    size = min(int(count), n)
    while True:
        block = rng.normal(size=(size,) + shape) + 1j * rng.normal(size=(size,) + shape)
        image = apply(block)
        for _ in range(2):
            image = apply(orthonormal(image))
        basis = orthonormal(image)
        mapped = apply(basis)
        flat_basis = basis.reshape(size, n)
        flat_mapped = mapped.reshape(size, n)
        reduced = flat_basis.conj() @ flat_mapped.T
        values, weights = np.linalg.eigh(0.5 * (reduced + reduced.conj().T))
        vectors = flat_basis.T @ weights
        residual = np.linalg.norm(flat_mapped.T @ weights - vectors * values[None, :], axis=0)
        order = np.argsort(-np.abs(values))
        values, vectors, residual = values[order], vectors[:, order], residual[order]
        top = np.abs(values[0])
        keep = np.abs(values) >= tolerance * top
        resolved = bool(np.abs(values[-1]) <= tolerance * top)
        accurate = bool(np.all(residual[keep] <= 1e-6 * top))
        if (resolved and accurate) or size >= min(max_states, n):
            return values, vectors, resolved and accurate
        size = min(2 * size, max_states, n)


def _propagate_flux_states(operators, eps, values, vectors, dt, max_time, norm_tolerance, check_every):
    """Split-operator propagation of thermal flux eigenstates under ``H0 - i eps``.

    Stops once the norm still on the grid can change the flux by less than
    ``norm_tolerance`` relative (``sum_k |f_k| N_k <= tol |sum_k f_k J_k|``).
    """
    shape = operators.shape
    batch = vectors.T.reshape((-1,) + shape).astype(complex)
    product = operators.product
    initial_product = np.sum(np.abs(batch) ** 2 * product, axis=tuple(range(1, batch.ndim)))
    propagators = []
    for matrix in operators.kinetic:
        t, u = eigh(matrix)
        propagators.append((u * np.exp(-1j * t * dt)[None, :]) @ u.T)
    eps = np.broadcast_to(eps, shape)
    half_factor = np.exp(-0.5j * dt * operators.potential - 0.5 * dt * eps)
    half_loss = 1.0 - np.exp(-dt * eps)
    absorbed_product = np.zeros(len(batch))
    absorbed_reactant = np.zeros(len(batch))
    axes = tuple(range(1, batch.ndim))
    history = [(0.0, absorbed_product.copy(), absorbed_reactant.copy(), np.ones(len(batch)), initial_product.copy())]
    steps = int(np.ceil(max_time / dt))
    time = 0.0
    for step in range(1, steps + 1):
        density = np.abs(batch) ** 2
        loss = density * half_loss
        absorbed_product += np.sum(loss * product, axis=axes)
        absorbed_reactant += np.sum(loss * (1.0 - product), axis=axes)
        batch = batch * half_factor
        for axis, matrix in enumerate(propagators):
            batch = operators.apply_axis(matrix, batch, axis)
        density = np.abs(batch) ** 2
        loss = density * half_loss
        absorbed_product += np.sum(loss * product, axis=axes)
        absorbed_reactant += np.sum(loss * (1.0 - product), axis=axes)
        batch = batch * half_factor
        time = step * dt
        if step % check_every == 0 or step == steps:
            density = np.abs(batch) ** 2
            remaining = np.sum(density, axis=axes)
            on_product = np.sum(density * product, axis=axes)
            history.append((time, absorbed_product.copy(), absorbed_reactant.copy(), remaining, on_product))
            current = abs(np.sum(values * (absorbed_product + on_product - initial_product)))
            if np.sum(np.abs(values) * remaining) <= norm_tolerance * max(current, 1e-300):
                break
            weight = np.abs(values)
            if len(history) >= 8 and np.sum(weight * (1.0 - remaining)) >= 0.9 * np.sum(weight):
                # Most of the flux has been absorbed: stop once the
                # branching-ratio extrapolation is stationary over the last
                # quarter and half of the run.
                estimate = _extrapolated_flux(history, len(history) - 1, values, initial_product)[0]
                scale = max(abs(estimate), 1e-300)
                stable = True
                for fraction, factor in ((0.75, 1.0), (0.5, 3.0)):
                    index = min(range(len(history)), key=lambda i: abs(history[i][0] - fraction * time))
                    other = _extrapolated_flux(history, index, values, initial_product)[0]
                    stable &= abs(other - estimate) <= factor * norm_tolerance * scale
                if stable:
                    break
    return history, initial_product


def _extrapolated_flux(history, index, values, initial_product):
    time, product, reactant, remaining, on_product = history[index]
    target = time / 2
    earlier = min(range(len(history)), key=lambda i: abs(history[i][0] - target))
    _, product0, reactant0, _, _ = history[earlier]
    gained_p = product - product0
    gained = gained_p + (reactant - reactant0)
    with np.errstate(divide="ignore", invalid="ignore"):
        branch = np.where(gained > 1e-14, gained_p / gained, np.where(remaining > 0, on_product / np.maximum(remaining, 1e-300), 0.0))
    total_product = product + branch * remaining
    return float(np.sum(values * (total_product - initial_product))), float(np.max(remaining))


def _iterative_rate(model, operators, eps, beta, shift, *, cutoff, flux_states, flux_tolerance,
                    max_flux_states, dt, max_time, norm_tolerance, chebyshev_tolerance):
    boltzmann = _Chebyshev(operators, shift, cutoff, 0.5 * beta, chebyshev_tolerance)
    values, vectors, flux_converged = _flux_eigenstates(operators, boltzmann, flux_states, flux_tolerance, max_flux_states)
    significant = np.abs(values) >= flux_tolerance * np.abs(values).max()
    values, vectors = values[significant], vectors[:, significant]
    check_every = max(1, int(round(max_time / dt / 400)))
    history, initial_product = _propagate_flux_states(operators, eps, values, vectors, dt, max_time,
                                                      norm_tolerance, check_every)
    final = len(history) - 1
    flux, trapped = _extrapolated_flux(history, final, values, initial_product)
    checks = {}
    total_time = history[final][0]
    for label, fraction in (("half", 0.5), ("three_quarter", 0.75), ("final", 1.0)):
        index = min(range(1, len(history)), key=lambda i: abs(history[i][0] - fraction * total_time))
        estimate, remaining = _extrapolated_flux(history, index, values, initial_product)
        checks[label] = {"time": float(history[index][0]), "flux": estimate, "max_remaining_norm": remaining}
    spread = max(abs(c["flux"] - flux) for c in checks.values()) / max(abs(flux), 1e-300)
    partition, info_q = _reactant_partition(model, beta, shift, operators=operators)
    return flux, partition, {
        "grid_points": operators.size,
        "chebyshev_order": boltzmann.order,
        "spectral_bounds": [boltzmann.emin, boltzmann.emax],
        "potential_cutoff": float(cutoff),
        "flux_eigenvalues": values.tolist(),
        "flux_states": int(len(values)),
        "flux_spectrum_converged": flux_converged,
        "time_step": float(dt),
        "propagated_time": float(total_time),
        "max_trapped_norm": trapped,
        "extrapolation": checks,
        "relative_extrapolation_spread": float(spread),
        "partition": info_q,
    }


def _truncation_threshold(model, truncation_energy):
    if callable(truncation_energy):
        values = np.asarray(truncation_energy(model.grid_points().reshape(-1, model.dimension)), dtype=float)
    else:
        values = np.asarray(truncation_energy, dtype=float)
    threshold = np.broadcast_to(values.reshape(model.shape) if values.size == np.prod(model.shape) else values,
                                model.shape)
    if not np.isfinite(threshold).all():
        raise ValueError("truncation_energy must be finite")
    return np.asarray(threshold, dtype=float)


def _truncation_margin(model, potential, threshold):
    """Threshold minus transverse-line floor at the reference and dividing-surface lines.

    ``truncation_margin`` is the smaller of the two (the lines that set the
    reactant normalization and the flux); ``truncation_margin_minimum`` is the
    smallest over all axis-0 lines that keep points, which also picks up box
    edges and is informative only for simple valleys.
    """
    if potential.ndim == 1:
        value = float(np.min(threshold - potential.min()))
        return {"truncation_margin": value, "truncation_margin_minimum": value}
    flat_v = potential.reshape(potential.shape[0], -1)
    flat_t = threshold.reshape(threshold.shape[0], -1)
    floor_index = flat_v.argmin(axis=1)
    rows = np.arange(potential.shape[0])
    margins = flat_t[rows, floor_index] - flat_v[rows, floor_index]
    kept = np.any(flat_v <= flat_t, axis=1)
    axis0 = model.axes[0]
    reference = int(np.argmin(abs(axis0 - model.reactant_reference)))
    surface = int(np.argmin(abs(axis0 - model.dividing_surface)))
    return {
        "truncation_margin": float(min(margins[reference], margins[surface])),
        "truncation_margin_reference": float(margins[reference]),
        "truncation_margin_dividing_surface": float(margins[surface]),
        "truncation_margin_minimum": float(margins[kept].min()) if kept.any() else None,
    }


def exact_quantum_rate(
    model: SubspaceModel,
    beta: float,
    *,
    method: str = "auto",
    dense_limit: int = 1600,
    cap_width: Any = None,
    cap_strength: Any = None,
    cap_power: int = 2,
    potential_cutoff: float | None = None,
    flux_states: int = 16,
    flux_tolerance: float = 1e-6,
    max_flux_states: int = 256,
    dt: float | None = None,
    max_time: float = 4.0e4,
    norm_tolerance: float = 1e-4,
    chebyshev_tolerance: float = 1e-14,
    absorber: Any = None,
    normalization: str = "default",
    energy_floor: float | None = None,
    truncation_energy: Any = None,
    memory_limit_mb: float | None = None,
) -> QuantumRateResult:
    """Exact thermal rate constant of ``model`` at inverse temperature ``beta``.

    ``method='auto'`` uses the dense Sylvester route up to ``dense_limit``
    grid points and thermal flux eigenstates beyond it.  The default
    absorbing potential occupies 20% of axis 0 at each end with a quadratic
    profile whose strength absorbs thermal wavepackets (``30 v_th / width``);
    both are numerical parameters to converge.  ``cap_width`` (and
    ``cap_strength``) may be a pair ``(reactant end, product end)``, e.g. for
    a product channel that ends in a well closer to the barrier than the
    reactant end is.  ``potential_cutoff``
    (iterative route) caps ``V`` in the Boltzmann operator only; the default
    is ``60/beta`` above the higher of the reactant reference and the lowest
    dividing-surface energy, so capped points carry less than ``exp(-30)`` of
    the thermal weight.

    Options for curved (e.g. hairpin) two-dimensional subspaces, dense route:

    ``absorber``
        callable ``eps(points[N, d])`` replacing the axis-0 absorbing ends,
        e.g. absorption keyed to arc length along a reaction path so that
        channels leaving the grid on the same side are both absorbed.
    ``normalization='slice'``
        divide by the quantum line density ``int <x|exp(-beta H)|x> dl`` on
        the axis-0 grid line nearest ``reactant_reference`` -- the exact
        counterpart of RPMD constrained TI along axis 0 when the reactant
        channel crosses that line obliquely.
    ``energy_floor``
        drop eigenstates below this energy from the thermal flux operator.
        The cumulative reaction probability vanishes below the higher
        channel threshold, so this is exact in the continuum limit; it
        removes deep states of the lower channel of an exo- or endothermic
        reaction whose zero net flux would otherwise be swamped by
        absorber reflection errors times an enormous Boltzmann weight.
    ``truncation_energy``
        energy-truncated DVR: grid points with ``V`` above it are removed.
        A number is one global threshold; an array of the grid shape, or a
        callable ``threshold(points[N, d])``, gives one threshold per point.
        The removed points act as hard walls that confine the transverse
        vibration; the rate converges only once the threshold lies about six
        quanta of the stiffest transverse mode above the valley floor
        everywhere along the channel, ridge included (walls at ~5
        ground-state widths).  The diagnostics report ``truncation_margin``:
        threshold minus transverse-line floor at the reactant reference and
        dividing-surface lines (the smaller), in energy units -- compare it
        with ``6 hbar omega`` of the stiffest transverse mode.
    """
    if not isinstance(model, SubspaceModel):
        raise TypeError("model must be a SubspaceModel")
    if model.metric is not None:
        raise ValueError("tensor-product exact rates do not support a position-dependent metric; "
                         "use the channel Hamiltonian interface")
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive")
    if method not in ("auto", "sylvester", "flux_eigenstates"):
        raise ValueError("method must be 'auto', 'sylvester' or 'flux_eigenstates'")
    operators = _GridOperators(model)
    axis0 = model.axes[0]
    length = float(axis0[-1] - axis0[0])
    widths = _pair(0.2 * length if cap_width is None else cap_width)
    if not all(0 < w < 0.5 * length for w in widths):
        raise ValueError("cap_width must be positive and below half the reaction-coordinate range")
    # A supplied absorber replaces the axis-0 end caps; it is checked on the
    # dividing surface itself below, not against widths it does not use.
    if absorber is None and not axis0[0] + widths[0] < model.dividing_surface < axis0[-1] - widths[1]:
        raise ValueError("the dividing surface must lie between the absorbing regions")
    velocity = np.sqrt(2.0 / (beta * model.masses[0]))
    strengths = tuple(30.0 * velocity / w for w in widths) if cap_strength is None else _pair(cap_strength)
    if not all(np.isfinite(v) and v > 0 for v in strengths):
        raise ValueError("cap_strength must be positive")
    # A scalar request is reported as a scalar, as before.
    width = widths[0] if widths[0] == widths[1] else list(widths)
    strength = strengths[0] if strengths[0] == strengths[1] else list(strengths)
    eps_axis = absorbing_potential(axis0, width=widths, strength=strengths, power=int(cap_power))
    eps = eps_axis.reshape((-1,) + (1,) * (model.dimension - 1))
    shift = float(operators.potential.min())
    if absorber is not None:
        values = absorber(model.grid_points().reshape(-1, model.dimension)) if callable(absorber) else absorber
        eps = np.asarray(values, dtype=float).reshape(model.shape)
        if not np.isfinite(eps).all() or np.any(eps < 0):
            raise ValueError("absorber must return finite nonnegative values")
        surface_line = int(np.argmin(np.abs(axis0 - model.dividing_surface)))
        if np.any(eps[surface_line] > 0):
            raise ValueError("the dividing surface must lie between the absorbing regions")
    if normalization not in ("default", "slice"):
        raise ValueError("normalization must be 'default' or 'slice'")
    keep = None
    margin_info = {}
    if truncation_energy is not None:
        threshold = _truncation_threshold(model, truncation_energy)
        keep = np.flatnonzero(operators.potential.ravel() <= threshold.ravel())
        margin_info = _truncation_margin(model, operators.potential, threshold)
    retained = operators.size if keep is None else len(keep)
    used = (keep is not None, energy_floor is not None, normalization != "default")
    plan = _route_plan(model, retained, method=method, dense_limit=dense_limit,
                       dense_only=[name for name, on in zip(_DENSE_ONLY_OPTIONS, used) if on],
                       normalization=normalization, max_flux_states=max_flux_states, memory_limit_mb=None)
    if not plan["feasible"]:
        # Refused before any propagation or factorisation: a structured diagnostic, not a generic error.
        raise DenseRouteUnavailable(plan)
    method = plan["route"]
    if method == "sylvester":
        estimate_mb = plan["dense_memory_mb"]
        if memory_limit_mb is not None and (not np.isfinite(memory_limit_mb) or memory_limit_mb <= 0):
            raise ValueError("memory_limit_mb must be finite and positive")
        if memory_limit_mb is not None and estimate_mb > memory_limit_mb:
            raise MemoryError(f"dense DVR needs an estimated {estimate_mb:.1f} MiB; budget is {memory_limit_mb:.1f} MiB")
        flux, partition, info = _dense_rate(model, operators, eps, beta, shift, keep=keep,
                                            energy_floor=energy_floor, normalization=normalization)
        info["estimated_peak_memory_mb"] = estimate_mb
    else:
        surface_index = int(np.searchsorted(axis0, model.dividing_surface))
        surface_energy = float(operators.potential[surface_index].min())
        reference_energy = float(np.min(model.energy(
            np.concatenate([np.full(operators.shape[1:] + (1,), model.reactant_reference),
                            np.stack(np.meshgrid(*model.axes[1:], indexing="ij"), axis=-1)], axis=-1)
            if model.dimension > 1 else np.array([[model.reactant_reference]]))))
        cutoff = (max(surface_energy, reference_energy) + 60.0 / beta) if potential_cutoff is None else float(potential_cutoff)
        energy_scale = max(surface_energy - shift, 0.0) + 10.0 / beta
        step = (0.05 * 2.0 * np.pi / energy_scale) if dt is None else float(dt)
        if not np.isfinite(step) or step <= 0 or not np.isfinite(max_time) or max_time <= step:
            raise ValueError("dt must be positive and smaller than max_time")
        flux, partition, info = _iterative_rate(
            model, operators, eps, beta, shift, cutoff=cutoff, flux_states=int(flux_states),
            flux_tolerance=float(flux_tolerance), max_flux_states=int(max_flux_states), dt=step,
            max_time=float(max_time), norm_tolerance=float(norm_tolerance),
            chebyshev_tolerance=float(chebyshev_tolerance),
        )
    if not np.isfinite(flux) or not np.isfinite(partition) or partition <= 0:
        raise FloatingPointError("exact quantum rate produced a non-finite flux or partition function")
    info.update(margin_info)
    info["solver_plan"] = plan
    info.update({
        "energy_shift": shift,
        "cap_width": width if absorber is None else None,
        "cap_strength": strength if absorber is None else None,
        "absorber": "axis-0 ends" if absorber is None else "callable",
        "cap_power": int(cap_power),
        "model": model.describe(),
        "units": "axis-0 length per time" if model.reactant == "asymptotic" else "per time",
    })
    return QuantumRateResult(beta, float(flux / partition), float(flux), float(partition), method, info)


dense_rate_matrix = _dense_rate_matrix

__all__ = ["DenseRouteUnavailable", "QuantumRateResult", "absorbing_potential", "exact_quantum_rate",
           "quantum_solver_plan", "sinc_dvr_kinetic", "dense_rate_matrix", "_dense_rate_matrix"]
