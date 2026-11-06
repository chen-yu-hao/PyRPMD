"""DVR system HEOM for a one-dimensional quantum coordinate.

The implementation follows the exponential-correlation HEOM convention

    C(t) = sum_k c_k exp(-nu_k t),

with ``hbar=1`` by default.  A hierarchy index is a tuple of non-negative
integers ``n=(n_0,...,n_{K-1})`` and the equations are

    d rho_n/dt = -i/hbar [H, rho_n] - (n.nu) rho_n
                  - i/hbar sum_k [Q, rho_{n+e_k}]
                  - i/hbar sum_k n_k (c_k Q rho_{n-e_k}
                                      - c_k.conj() rho_{n-e_k} Q).

The finite hierarchy is cut at ``sum(n) <= depth`` by dropping upward terms
from the top tier.  The class is deliberately small and dense: it is intended
for low-dimensional validation calculations and for an unambiguous reference
implementation, rather than production large-system HEOM.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

ArrayLike = np.ndarray | Sequence[float]


def _as_operator(value: ArrayLike | Callable[[np.ndarray], ArrayLike], grid: np.ndarray, name: str) -> np.ndarray:
    """Evaluate a scalar function/array on a grid and return a finite vector."""
    if callable(value):
        out = np.asarray(value(grid))
    else:
        out = np.asarray(value)
    if out.ndim == 0:
        out = np.full(grid.shape, out, dtype=out.dtype)
    out = np.asarray(out, dtype=float)
    if out.shape != grid.shape:
        raise ValueError(f"{name} must have shape {grid.shape}, got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _validate_exponents(exponents: Iterable[tuple[complex, complex]]) -> tuple[np.ndarray, np.ndarray]:
    pairs = list(exponents)
    if not pairs:
        raise ValueError("at least one correlation exponent is required")
    coeff = np.asarray([p[0] for p in pairs], dtype=complex)
    rates = np.asarray([p[1] for p in pairs], dtype=complex)
    if coeff.ndim != 1 or rates.ndim != 1 or coeff.size != rates.size:
        raise ValueError("exponents must be (coefficient, rate) pairs")
    if not np.all(np.isfinite(coeff)) or not np.all(np.isfinite(rates)):
        raise ValueError("correlation coefficients and rates must be finite")
    if np.any(np.real(rates) <= 0):
        raise ValueError("all correlation rates must have positive real part")
    return coeff, rates


def _hierarchy_indices(n_exponents: int, depth: int) -> tuple[tuple[int, ...], ...]:
    if n_exponents < 1:
        raise ValueError("n_exponents must be positive")
    if int(depth) != depth or depth < 0:
        raise ValueError("depth must be a non-negative integer")
    depth = int(depth)
    # Lexicographic order is deterministic and makes output reproducible.
    indices = [idx for idx in product(range(depth + 1), repeat=n_exponents) if sum(idx) <= depth]
    indices.sort(key=lambda x: (sum(x), x))
    return tuple(indices)


@dataclass
class DVRHEOMResult:
    """Result returned by :meth:`DVRSystemHEOM.propagate`."""

    times: np.ndarray
    hierarchy: np.ndarray
    success: bool = True
    message: str = ""

    @property
    def rho(self) -> np.ndarray:
        """Root density matrices with shape ``(nt, dim, dim)``."""
        return self.hierarchy[:, 0]

    @property
    def root(self) -> np.ndarray:
        return self.rho

    def expect(self, operator: np.ndarray) -> np.ndarray:
        """Return ``Tr(operator rho(t))`` for the root ADO."""
        op = np.asarray(operator)
        if op.shape != self.rho.shape[-2:]:
            raise ValueError("operator shape does not match density matrices")
        return np.einsum("ij,tji->t", op, self.rho)

    @property
    def trace_error(self) -> float:
        """Maximum deviation of the root trace from its initial value."""
        traces = np.trace(self.rho, axis1=1, axis2=2)
        return float(np.max(np.abs(traces - traces[0])))

    @property
    def hermiticity_error(self) -> float:
        """Maximum Frobenius anti-Hermitian norm of the root ADO."""
        anti = self.rho - np.swapaxes(self.rho.conj(), -1, -2)
        return float(np.max(np.linalg.norm(anti, axis=(-2, -1))))


class DVRSystemHEOM:
    """Dense 1-D DVR HEOM solver with a multi-exponential bath correlation.

    Parameters
    ----------
    grid:
        One-dimensional, uniformly spaced DVR coordinate grid.
    potential, coupling:
        Callable evaluated on ``grid`` or arrays of length ``len(grid)``.
        The system Hamiltonian is ``T + V(grid) + counterterm*f(grid)**2``;
        the system-bath coupling operator is ``Q=f(grid)``.
    beta:
        Inverse temperature.  It is stored for provenance and validation; the
        supplied exponential coefficients already define the quantum
        correlation function.
    exponents:
        Sequence of ``(c_k, nu_k)`` pairs for ``C(t)=sum c_k exp(-nu_k t)``.
    counterterm:
        ``None``/zero, a scalar coefficient multiplying ``f**2``, or a
        callable/array giving an additive potential on the grid.
    hamiltonian:
        Optional explicit dense system Hamiltonian.  If given, it takes
        precedence over the finite-difference kinetic plus potential.
    """

    def __init__(
        self,
        grid: ArrayLike,
        potential: ArrayLike | Callable[[np.ndarray], ArrayLike],
        coupling: ArrayLike | Callable[[np.ndarray], ArrayLike],
        beta: float,
        exponents: Iterable[tuple[complex, complex]],
        *,
        mass: float = 1.0,
        hbar: float = 1.0,
        counterterm: float | ArrayLike | Callable[[np.ndarray], ArrayLike] | None = None,
        depth: int = 4,
        hamiltonian: np.ndarray | None = None,
    ) -> None:
        self.grid = np.asarray(grid, dtype=float)
        if self.grid.ndim != 1 or self.grid.size < 2 or not np.all(np.isfinite(self.grid)):
            raise ValueError("grid must be a finite one-dimensional array with at least two points")
        dx = np.diff(self.grid)
        if not np.all(dx > 0) or not np.allclose(dx, dx[0], rtol=1e-10, atol=1e-13):
            raise ValueError("grid must be strictly increasing and uniformly spaced")
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be positive and finite")
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("mass must be positive and finite")
        if not np.isfinite(hbar) or hbar <= 0:
            raise ValueError("hbar must be positive and finite")
        self.beta = float(beta)
        self.mass = float(mass)
        self.hbar = float(hbar)
        if isinstance(depth, (bool, np.bool_)) or int(depth) != depth or depth < 0:
            raise ValueError("depth must be a non-negative integer")
        self.depth = int(depth)
        self.potential_values = _as_operator(potential, self.grid, "potential")
        self.coupling_values = _as_operator(coupling, self.grid, "coupling")
        self.coefficients, self.rates = _validate_exponents(exponents)
        self.n_exponents = self.coefficients.size
        self.indices = _hierarchy_indices(self.n_exponents, self.depth)
        self.index_map = {idx: i for i, idx in enumerate(self.indices)}
        self.dim = self.grid.size
        self.Q = np.diag(self.coupling_values.astype(complex))

        if hamiltonian is not None:
            H = np.asarray(hamiltonian, dtype=complex)
            if H.shape != (self.dim, self.dim) or not np.all(np.isfinite(H)):
                raise ValueError("hamiltonian must be a finite square matrix matching grid")
            self.H = 0.5 * (H + H.conj().T)
            self.kinetic = self.H - np.diag(self.potential_values)
        else:
            dx0 = dx[0]
            main = np.full(self.dim, self.hbar**2 / (self.mass * dx0**2), dtype=float)
            off = np.full(self.dim - 1, -self.hbar**2 / (2.0 * self.mass * dx0**2), dtype=float)
            T = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)
            if counterterm is None:
                ct = np.zeros(self.dim)
            elif callable(counterterm) or np.ndim(counterterm) > 0:
                # An array/function is an explicit additive potential.
                ct = _as_operator(counterterm, self.grid, "counterterm")  # type: ignore[arg-type]
            else:
                ct = float(counterterm) * self.coupling_values**2
            self.counterterm_values = ct
            self.kinetic = T.astype(complex)
            self.H = T.astype(complex) + np.diag(self.potential_values + ct)
        if not hasattr(self, "counterterm_values"):
            self.counterterm_values = np.zeros(self.dim)
        self._shape = (len(self.indices), self.dim, self.dim)

    @property
    def hierarchy_size(self) -> int:
        return len(self.indices)

    @property
    def n_ados(self) -> int:
        """Number of auxiliary density operators in the truncated hierarchy."""
        return self.hierarchy_size

    @property
    def size(self) -> int:
        return self.hierarchy_size

    @property
    def hierarchy_indices(self) -> tuple[tuple[int, ...], ...]:
        """Deterministic list of hierarchy multi-indices."""
        return self.indices

    @property
    def system_hamiltonian(self) -> np.ndarray:
        """Dense system Hamiltonian (alias useful in diagnostics)."""
        return self.H

    @property
    def coupling_operator(self) -> np.ndarray:
        """Dense system coupling operator ``Q``."""
        return self.Q

    def correlation(self, times: Iterable[float]) -> np.ndarray:
        """Evaluate the fitted bath correlation ``C(t)``."""
        t = np.asarray(times, dtype=float)
        if np.any(~np.isfinite(t)) or np.any(t < 0):
            raise ValueError("times must be finite and non-negative")
        return np.exp(-np.outer(t, self.rates)) @ self.coefficients

    def initial_hierarchy(self, rho0: np.ndarray) -> np.ndarray:
        """Return a hierarchy with ``rho0`` at the root and zero ADOs."""
        rho = np.asarray(rho0, dtype=complex)
        if rho.shape != (self.dim, self.dim):
            raise ValueError(f"rho0 must have shape {(self.dim, self.dim)}")
        out = np.zeros(self._shape, dtype=complex)
        out[0] = rho
        return out

    def _rhs_hierarchy(self, hierarchy: np.ndarray) -> np.ndarray:
        out = np.zeros_like(hierarchy, dtype=complex)
        ih = 1j / self.hbar
        for i, idx in enumerate(self.indices):
            rho = hierarchy[i]
            out[i] += -ih * (self.H @ rho - rho @ self.H)
            out[i] -= np.dot(idx, self.rates) * rho
            for k in range(self.n_exponents):
                up = list(idx)
                up[k] += 1
                up_t = tuple(up)
                j = self.index_map.get(up_t)
                if j is not None:
                    out[i] += -ih * (self.Q @ hierarchy[j] - hierarchy[j] @ self.Q)
                n = idx[k]
                if n:
                    down = list(idx)
                    down[k] -= 1
                    jdown = self.index_map[tuple(down)]
                    lower = hierarchy[jdown]
                    c = self.coefficients[k]
                    out[i] += -ih * n * (c * self.Q @ lower - np.conj(c) * lower @ self.Q)
        return out

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Right-hand side suitable for :func:`scipy.integrate.solve_ivp`."""
        del t
        arr = np.asarray(y, dtype=complex)
        flat = arr.ndim == 1
        if flat:
            if arr.size != np.prod(self._shape):
                raise ValueError("flattened state has the wrong size")
            arr = arr.reshape(self._shape)
        if arr.shape != self._shape:
            raise ValueError(f"state must have shape {self._shape}")
        out = self._rhs_hierarchy(arr)
        return out.ravel() if flat else out

    def propagate(
        self,
        times: Iterable[float],
        rho0: np.ndarray | Iterable[float] | None = None,
        *,
        rtol: float = 1e-9,
        atol: float = 1e-11,
        method: str = "DOP853",
        **solve_kwargs,
    ) -> DVRHEOMResult:
        """Propagate a hierarchy on an increasing output-time grid."""
        if rho0 is None:
            raise TypeError("propagate requires both times and rho0")
        # Accept the common ``propagate(rho0, times)`` spelling as well as the
        # documented ``propagate(times, rho0)``.  The dimensions are
        # unambiguous for density matrices versus one-dimensional time grids.
        first = np.asarray(times)
        second = np.asarray(rho0)
        if first.ndim == 2 and second.ndim == 1:
            times, rho0 = rho0, times  # type: ignore[assignment]
        t_eval = np.asarray(times, dtype=float)
        if t_eval.ndim != 1 or t_eval.size < 1 or np.any(~np.isfinite(t_eval)):
            raise ValueError("times must be a finite one-dimensional array")
        if t_eval.size > 1 and np.any(np.diff(t_eval) <= 0):
            raise ValueError("times must be strictly increasing")
        if t_eval[0] < 0:
            raise ValueError("times must be non-negative")
        initial = self.initial_hierarchy(rho0).ravel()
        if t_eval.size == 1 and t_eval[0] == 0:
            return DVRHEOMResult(t_eval, initial.reshape((1,) + self._shape))
        t0 = float(t_eval[0])
        if t0 != 0.0:
            # Integrate from zero so the input state always has a clear meaning.
            full_eval = np.concatenate(([0.0], t_eval))
            sol = solve_ivp(self.rhs, (0.0, float(t_eval[-1])), initial, t_eval=full_eval,
                            method=method, rtol=rtol, atol=atol, **solve_kwargs)
            values = sol.y[:, 1:].T
        else:
            sol = solve_ivp(self.rhs, (0.0, float(t_eval[-1])), initial, t_eval=t_eval,
                            method=method, rtol=rtol, atol=atol, **solve_kwargs)
            values = sol.y.T
        hierarchy = values.reshape((t_eval.size,) + self._shape)
        return DVRHEOMResult(t_eval, hierarchy, bool(sol.success), str(sol.message))

    run = propagate
    evolve = propagate

    def equilibrium_state(self) -> np.ndarray:
        """Return the isolated-system canonical state for diagnostics."""
        evals, evecs = np.linalg.eigh(self.H)
        weights = np.exp(-self.beta * (evals - evals.min()))
        rho = (evecs * weights) @ evecs.conj().T
        return rho / np.trace(rho)

    equilibrium = equilibrium_state

    def steady_state(
        self,
        rho0: np.ndarray | None = None,
        *,
        t_max: float = 100.0,
        n_steps: int = 1001,
        tol: float = 1e-8,
        rtol: float = 1e-9,
        atol: float = 1e-11,
        return_result: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, HEOMResult]:
        """Estimate a stationary reduced state by a fixed long-time run.

        This deliberately exposes the propagation horizon and convergence
        tolerance so a validation record can report them.  A stationary state
        is not unique for a zero-coupling/unitary model; in that case the
        method raises instead of silently returning a transient matrix.
        """
        if t_max <= 0 or n_steps < 2 or tol <= 0:
            raise ValueError("t_max, n_steps and tol must be positive")
        if rho0 is None:
            rho0 = self.equilibrium_state()
        times = np.linspace(0.0, float(t_max), int(n_steps))
        result = self.propagate(times, rho0, rtol=rtol, atol=atol)
        change = float(np.linalg.norm(result.rho[-1] - result.rho[-2]))
        if not result.success or change > tol:
            raise RuntimeError(
                f"HEOM steady-state run did not converge: final change={change:.3e}, tol={tol:.3e}"
            )
        state = result.rho[-1]
        state = 0.5 * (state + state.conj().T)
        state = state / np.trace(state)
        return (state, result) if return_result else state



# Compatibility alias for callers that expect the generic HEOM result name.
HEOMResult = DVRHEOMResult


def drude_lorentz_exponents(
    beta: float,
    reorganization: float,
    cutoff: float,
    *,
    matsubara_terms: int = 0,
    hbar: float = 1.0,
) -> tuple[tuple[complex, complex], ...]:
    """Return a Matsubara exponential decomposition for a Drude bath.

    The convention is ``J(w)=2*lambda*gamma*w/(w**2+gamma**2)`` and
    ``C(t)=sum c_k exp(-nu_k*t)``.  The decomposition is explicit so the
    caller can record the chosen Matsubara cutoff as part of convergence data.
    """
    beta = float(beta)
    lam = float(reorganization)
    gamma = float(cutoff)
    if beta <= 0 or lam < 0 or gamma <= 0 or hbar <= 0:
        raise ValueError("beta, cutoff and hbar must be positive; reorganization must be non-negative")
    if int(matsubara_terms) != matsubara_terms or matsubara_terms < 0:
        raise ValueError("matsubara_terms must be a non-negative integer")
    x = beta * hbar * gamma / 2.0
    if abs(np.sin(x)) < 1e-12:
        raise ValueError("Drude zero-term cotangent is singular at beta*hbar*gamma/2 = n*pi")
    terms: list[tuple[complex, complex]] = [(lam * gamma * (1.0 / np.tan(x) - 1j), gamma)]
    for n in range(1, int(matsubara_terms) + 1):
        nu = 2.0 * np.pi * n / (beta * hbar)
        if abs(nu**2 - gamma**2) < 1e-12 * max(1.0, gamma**2):
            raise ValueError("Matsubara frequency nearly coincides with Drude cutoff")
        coeff = (4.0 * lam * gamma / (beta * hbar)) * nu / (nu**2 - gamma**2)
        terms.append((coeff, nu))
    return tuple(terms)


def drude_lorentz_spectral_density(
    frequencies: Iterable[float], reorganization: float, cutoff: float
) -> np.ndarray:
    """Evaluate ``J(w)=2 lambda gamma w/(w**2+gamma**2)`` for ``w>=0``."""
    w = np.asarray(frequencies, dtype=float)
    if w.ndim != 1 or np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("frequencies must be a finite non-negative vector")
    if reorganization < 0 or cutoff <= 0:
        raise ValueError("reorganization must be non-negative and cutoff positive")
    return 2.0 * float(reorganization) * float(cutoff) * w / (w**2 + float(cutoff)**2)


__all__ = [
    "DVRSystemHEOM", "DVRHEOMResult", "HEOMResult",
    "drude_lorentz_exponents", "drude_lorentz_spectral_density",
]
