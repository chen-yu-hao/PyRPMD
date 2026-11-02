"""Centroid adaptive biasing forces and extended-system ABF in atomic units.

The target is the *CV probability density*, including its geometrical Jacobian.
For centroid gradient G, mass matrix M, and dual field
``v = (G M^-1 G.T)^-1 G M^-1``, the local free-energy gradient is
``-v @ mean_bead_force - div(v)/beta``. Ring springs cancel under a common
centroid displacement. No rate/flux-measure correction is applied.

The applied bias is the conservative grid projection of the accumulated mean
force (projected ABF); its grid and update interval are scientific parameters.
eABF biases an auxiliary coordinate, and reports the physical CV surface using
the corrected z-averaged restraint (CZAR) estimator, not its smoothed surface.
See Darve et al., JCP 128, 144120 (2008), doi:10.1063/1.2829861; Lesage et al.,
JPCB 121, 3676 (2017), doi:10.1021/acs.jpcb.6b10055.
"""

from __future__ import annotations

from itertools import product
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsmr

from .analysis.fes_result import FESResult
from .cv import Cartesian, CVStack, Distance, COMDistance
from .validation import checked_integer, finite_positive


def _array(value, name, shape=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite real numbers")
    result = np.array(raw, dtype=np.float64, copy=True)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result


def validate_centroid_cv(cv):
    """Require an explicit nonperiodic centroid contract, never infer one."""
    if getattr(cv, "dimension", None) not in (1, 2):
        raise ValueError("ABF supports one or two CV dimensions")
    components = cv.cvs if isinstance(cv, CVStack) else (cv,)
    if any(getattr(c, "estimator", None) != "centroid" for c in components):
        raise ValueError("ABF requires explicitly declared centroid CVs")
    if len(cv.periods) != cv.dimension or any(p is not None for p in cv.periods):
        raise ValueError("ABF currently requires nonperiodic CVs")
    return cv


def _centroid_geometry(cv, q):
    q = _array(q, "coordinates")
    if q.ndim != 3 or q.shape[-1] != 3 or min(q.shape) < 1:
        raise ValueError("coordinates must have nonempty [P,N,3] shape")
    result = cv.value_and_jacobian(q)
    value = _array(result.value, "CV values", (cv.dimension,))
    jac = _array(result.jacobian, "CV Jacobian", (cv.dimension,) + q.shape)
    g = jac.sum(axis=1)
    if not np.allclose(jac, g[:, None] / len(q), rtol=1e-11, atol=1e-13):
        raise ValueError("centroid CV Jacobian must be identical on every bead")
    return value, g.reshape(cv.dimension, -1), jac


def _dual(gradient, inverse_mass):
    weighted = gradient * inverse_mass
    metric = weighted @ gradient.T
    eigen = np.linalg.eigvalsh(metric)
    if eigen[0] <= 0 or eigen[0] <= 1e-12 * eigen[-1] or not np.isfinite(eigen).all():
        raise ValueError("ABF CV mass metric is singular or ill-conditioned")
    return np.linalg.solve(metric, weighted)


class MeanForceEstimator:
    """Unbiased local gradient estimator, including the Jacobian/Fixman term.

    ``derivative='analytic'`` supports unmodified Cartesian coordinates and a
    single Distance/COMDistance. ``finite_difference`` explicitly differentiates
    the dual field using centroid displacements; it stores no dense Hessian,
    but costs 2*3N Jacobian evaluations per observation. It is never selected
    as a silent fallback. Forces must be unbiased physical PES forces [P,N,3].
    """

    def __init__(self, cv, masses, beta, *, derivative="analytic", fd_step=1e-5):
        self.cv = validate_centroid_cv(cv)
        self.masses = _array(masses, "masses")
        if self.masses.ndim != 1 or not len(self.masses) or np.any(self.masses <= 0):
            raise ValueError("masses must be a positive vector")
        self.beta = finite_positive(beta, "beta")
        self.fd_step = finite_positive(fd_step, "fd_step")
        self.derivative = derivative
        components = cv.cvs if isinstance(cv, CVStack) else (cv,)
        self._cartesian = all(type(c) is Cartesian for c in components)
        self._radial = (len(components) == 1 and type(components[0]) in (Distance, COMDistance))
        self._component = components[0]
        if derivative not in ("analytic", "finite_difference"):
            raise ValueError("ABF derivative must be analytic or finite_difference")
        if derivative == "analytic":
            from .cv import _BUILTIN_CV_METHODS
            safe = all(not any(name in c.__dict__ or getattr(type(c), name) is not original
                               for name, original in _BUILTIN_CV_METHODS[type(c)].items())
                       for c in components if type(c) in _BUILTIN_CV_METHODS)
            if isinstance(cv, CVStack):
                safe &= type(cv) is CVStack and all(
                    name not in cv.__dict__ and getattr(type(cv), name) is original
                    for name, original in _BUILTIN_CV_METHODS[CVStack].items())
            if not (self._cartesian or self._radial) or not safe:
                raise ValueError("analytic ABF supports unmodified Cartesian or one Distance/COMDistance; "
                                 "select derivative='finite_difference' explicitly for other centroid CVs")
        self._inverse_mass = np.repeat(1.0 / self.masses, 3)

    def evaluate(self, q, forces):
        q = _array(q, "coordinates")
        if q.ndim != 3 or q.shape[1:] != (len(self.masses), 3):
            raise ValueError("coordinates disagree with ABF masses")
        forces = _array(forces, "physical forces", q.shape)
        value, gradient, _ = _centroid_geometry(self.cv, q)
        dual = _dual(gradient, self._inverse_mass)
        if self.derivative == "analytic":
            divergence = np.zeros(self.cv.dimension) if self._cartesian else np.array([2.0 / value[0]])
        else:
            # All bead positions move together: this differentiates in
            # centroid coordinates and does not introduce an extra P factor.
            divergence = np.zeros(self.cv.dimension)
            centroid = q.mean(axis=0).ravel()
            for coordinate in range(gradient.shape[1]):
                h = self.fd_step * max(1.0, abs(centroid[coordinate]))
                atom, axis = divmod(coordinate, 3)
                plus, minus = q.copy(), q.copy()
                plus[:, atom, axis] += h
                minus[:, atom, axis] -= h
                gp = _centroid_geometry(self.cv, plus)[1]
                gm = _centroid_geometry(self.cv, minus)[1]
                divergence += (_dual(gp, self._inverse_mass)[:, coordinate]
                               - _dual(gm, self._inverse_mass)[:, coordinate]) / (2 * h)
        estimate = -dual @ forces.mean(axis=0).ravel() - divergence / self.beta
        if not np.isfinite(estimate).all():
            raise FloatingPointError("nonfinite ABF local mean force")
        return value, estimate


class ForceGrid:
    """Bounded-memory Welford statistics on a one- or two-dimensional grid."""

    def __init__(self, edges):
        self.edges = tuple(_array(e, "grid edges") for e in edges)
        if len(self.edges) not in (1, 2):
            raise ValueError("ABF grid needs one or two axes")
        if any(e.ndim != 1 or len(e) < 3 or np.any(np.diff(e) <= 0) for e in self.edges):
            raise ValueError("ABF edges must increase and contain at least two bins")
        self.dimension = len(self.edges)
        self.shape = tuple(len(e) - 1 for e in self.edges)
        if np.prod(self.shape) > 100_000:
            raise ValueError("ABF grid exceeds 100000 bins")
        self.count = np.zeros(self.shape, dtype=np.int64)
        self.mean = np.zeros(self.shape + (self.dimension,))
        self.m2 = np.zeros_like(self.mean)
        self.outside = 0

    def index(self, value):
        value = _array(value, "CV values", (self.dimension,))
        if any(x < e[0] or x > e[-1] for x, e in zip(value, self.edges)):
            return None
        return tuple(min(len(e) - 2, np.searchsorted(e, x, side="right") - 1)
                     for x, e in zip(value, self.edges))

    def update(self, value, gradient):
        force = _array(gradient, "free-energy gradient", (self.dimension,))
        index = self.index(value)
        if index is None:
            self.outside += 1
            return
        if self.count[index] == np.iinfo(np.int64).max:
            raise OverflowError("ABF observation clock overflow")
        self.count[index] += 1
        delta = force - self.mean[index]
        mean = self.mean[index] + delta / self.count[index]
        m2 = self.m2[index] + delta * (force - mean)
        if not np.isfinite(mean).all() or not np.isfinite(m2).all():
            self.count[index] -= 1
            raise FloatingPointError("ABF force statistics overflow")
        self.mean[index], self.m2[index] = mean, m2

    @property
    def variance(self):
        return np.divide(self.m2, (self.count - 1)[..., None],
                         out=np.full_like(self.mean, np.nan), where=(self.count > 1)[..., None])

    def state_dict(self):
        return {"edges": [e.tolist() for e in self.edges], "count": self.count.tolist(),
                "mean": self.mean.tolist(), "m2": self.m2.tolist(), "outside": self.outside}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"edges", "count", "mean", "m2", "outside"}:
            raise ValueError("invalid ABF grid checkpoint")
        if state["edges"] != [e.tolist() for e in self.edges]:
            raise ValueError("ABF checkpoint grid differs")
        raw = np.asarray(state["count"])
        if raw.shape != self.shape or raw.dtype.kind not in "iu" or np.any(raw < 0):
            raise ValueError("invalid ABF checkpoint counts")
        if np.any(raw > np.iinfo(np.int64).max):
            raise ValueError("ABF checkpoint count overflow")
        mean = _array(state["mean"], "ABF checkpoint mean", self.mean.shape)
        m2 = _array(state["m2"], "ABF checkpoint m2", self.m2.shape)
        if np.any(m2 < -1e-12) or np.any(mean[raw == 0]) or np.any(m2[raw < 2]):
            raise ValueError("ABF checkpoint statistics disagree with counts")
        outside = checked_integer(state["outside"], "ABF outside count", 0)
        self.count, self.mean, self.m2, self.outside = raw.astype(np.int64), mean, m2, outside


def integrate_mean_force(edges, gradient, *, counts=None, min_count=1):
    """Trapezoidal graph integration with a single connected reference gauge.

    In 2D, solve the weighted least-squares edge differences (a discrete
    Helmholtz projection). Disconnected islands cannot share a free-energy
    offset: retain only the largest component, and mask the others explicitly.
    Returns (surface, mask, diagnostics); no count-based SEM is invented.
    """
    grid = ForceGrid(edges)
    force = _array(gradient, "mean force", grid.mean.shape)
    minimum = checked_integer(min_count, "min_count", 0)
    count = np.ones(grid.shape) if counts is None else _array(counts, "counts", grid.shape)
    if np.any(count < 0):
        raise ValueError("counts must be nonnegative")
    support = count >= minimum
    indices = np.arange(np.prod(grid.shape)).reshape(grid.shape)
    left, right, delta, weight = [], [], [], []
    centers = [(e[1:] + e[:-1]) * 0.5 for e in grid.edges]
    for axis in range(grid.dimension):
        a = [slice(None)] * grid.dimension
        b = a.copy()
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        valid = support[a] & support[b]
        spacing = np.diff(centers[axis]).reshape(tuple(-1 if j == axis else 1 for j in range(grid.dimension)))
        differences = 0.5 * (force[a][..., axis] + force[b][..., axis]) * spacing
        left.extend(indices[a][valid].tolist())
        right.extend(indices[b][valid].tolist())
        delta.extend(differences[valid].tolist())
        # Equal weights remain defined for empty cells in the *bias* projection.
        ca, cb = np.maximum(count[a][valid], 1), np.maximum(count[b][valid], 1)
        weight.extend(np.sqrt(2 * ca * cb / (ca + cb)).tolist())
    n = indices.size
    adjacency = sparse.coo_matrix((np.ones(2 * len(left)), (left + right, right + left)), shape=(n, n)).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    labels_supported, sizes = np.unique(labels[support.ravel()], return_counts=True)
    result = np.full(grid.shape, np.nan)
    if not len(sizes):
        return result, np.ones(grid.shape, bool), {"components": 0, "discarded_bins": 0, "edge_residual_rms": None}
    chosen = labels_supported[np.argmax(sizes)]
    kept = support.ravel() & (labels == chosen)
    cells = np.flatnonzero(kept)
    mapping = np.full(n, -1, int)
    mapping[cells] = np.arange(len(cells))
    left, right = np.asarray(left, int), np.asarray(right, int)
    use = kept[left] & kept[right]
    a, b = mapping[left[use]], mapping[right[use]]
    target, weights = np.asarray(delta)[use], np.asarray(weight)[use]
    if len(cells) == 1:
        solution, residual = np.zeros(1), 0.0
    elif grid.dimension == 1:
        solution = np.r_[0.0, np.cumsum(target)]
        residual = 0.0
    else:
        weights /= weights.max()
        rows = np.arange(len(a))
        matrix = sparse.coo_matrix((np.r_[-weights, weights, 1.0],
                                   (np.r_[rows, rows, len(a)], np.r_[a, b, 0])),
                                  shape=(len(a) + 1, len(cells))).tocsr()
        fitted = lsmr(matrix, np.r_[weights * target, 0.0], atol=1e-12, btol=1e-12,
                      maxiter=max(1000, 4 * len(cells)))
        if fitted[1] not in (0, 1, 2, 4, 5):
            raise RuntimeError("ABF grid projection did not converge")
        solution = fitted[0]
        solution -= solution[0]
        residual = float(np.sqrt(np.mean((solution[b] - solution[a] - target) ** 2)))
    result.ravel()[cells] = solution
    return result, ~kept.reshape(grid.shape), {
        "components": len(sizes), "discarded_bins": int(support.sum() - len(cells)),
        "edge_residual_rms": residual, "reference_bin": list(np.unravel_index(cells[0], grid.shape)),
    }


def _interpolate(surface, edges, value):
    """Multilinear scalar interpolation and its exact analytic gradient."""
    dimension = len(edges)
    lo, t, spacing = [], [], []
    for axis, edge in enumerate(edges):
        centers = 0.5 * (edge[1:] + edge[:-1])
        i = int(np.clip(np.searchsorted(centers, value[axis]) - 1, 0, len(centers) - 2))
        dx = centers[i + 1] - centers[i]
        lo.append(i)
        t.append((value[axis] - centers[i]) / dx)
        spacing.append(dx)
    energy, gradient = 0.0, np.zeros(dimension)
    for corner in product((0, 1), repeat=dimension):
        weights = [t[d] if corner[d] else 1.0 - t[d] for d in range(dimension)]
        point = float(surface[tuple(lo[d] + corner[d] for d in range(dimension))])
        energy += point * np.prod(weights)
        for d in range(dimension):
            gradient[d] += point * (1.0 if corner[d] else -1.0) / spacing[d] * np.prod(
                [weights[j] for j in range(dimension) if j != d])
    return energy, gradient


class ABFBias:
    """Conservative projected ABF, with physical W and H_P += P*W.

    Adaptation is explicit in ``observe``/``refresh``. Force evaluation never
    changes history. Outside the grid the adaptive energy is held fixed in
    the outward direction, and a quadratic wall confines each coordinate.
    """

    def __init__(self, cv, edges, beta, *, full_samples=200, wall_stiffness=1.0):
        self.cv = validate_centroid_cv(cv)
        self.grid = ForceGrid(edges)
        if self.grid.dimension != cv.dimension:
            raise ValueError("ABF grid/CV dimension mismatch")
        self.beta = finite_positive(beta, "beta")
        self.full_samples = checked_integer(full_samples, "full_samples", 1)
        self.wall_stiffness = _array(np.broadcast_to(wall_stiffness, (cv.dimension,)), "wall_stiffness")
        if np.any(self.wall_stiffness <= 0):
            raise ValueError("wall_stiffness must be positive")
        self.surface = np.zeros(self.grid.shape)
        self.observations = 0
        self.refreshes = 0

    def observe(self, value, gradient):
        self.grid.update(value, gradient)
        self.observations += 1

    def refresh(self):
        ramp = np.minimum(self.grid.count / self.full_samples, 1.0)
        self.surface, _, _ = integrate_mean_force(self.grid.edges, self.grid.mean * ramp[..., None], min_count=0)
        self.refreshes += 1

    def physical_energy_gradient(self, value):
        value = _array(value, "CV values", (self.cv.dimension,))
        lower = np.array([e[0] for e in self.grid.edges])
        upper = np.array([e[-1] for e in self.grid.edges])
        clipped = np.clip(value, lower, upper)
        energy, gradient = _interpolate(self.surface, self.grid.edges, clipped)
        delta = value - clipped
        gradient[delta != 0] = 0.0
        return (-energy + 0.5 * np.dot(self.wall_stiffness, delta * delta),
                -gradient + self.wall_stiffness * delta)

    def physical_energy(self, value):
        return self.physical_energy_gradient(value)[0]

    def energy_force(self, q):
        value, _, jacobian = _centroid_geometry(self.cv, q)
        energy, gradient = self.physical_energy_gradient(value)
        return len(q) * energy, -len(q) * np.einsum("d,dpnc->pnc", gradient, jacobian)

    def state_dict(self):
        return {"schema": 1, "method": "abf", "beta": self.beta,
                "full_samples": self.full_samples, "wall_stiffness": self.wall_stiffness.tolist(),
                "grid": self.grid.state_dict(), "surface": self.surface.tolist(),
                "observations": self.observations, "refreshes": self.refreshes}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError("invalid ABF checkpoint fields")
        for key in ("schema", "method", "beta", "full_samples", "wall_stiffness"):
            if state[key] != expected[key]:
                raise ValueError("ABF checkpoint identity differs: " + key)
        grid = ForceGrid(self.grid.edges)
        grid.load_state_dict(state["grid"])
        surface = _array(state["surface"], "ABF checkpoint surface", self.grid.shape)
        observations = checked_integer(state["observations"], "ABF observations", 0)
        refreshes = checked_integer(state["refreshes"], "ABF refreshes", 0)
        if observations != int(grid.count.sum()) + grid.outside or refreshes > observations:
            raise ValueError("ABF checkpoint observation clock disagrees with statistics")
        self.grid, self.surface = grid, surface
        self.observations, self.refreshes = observations, refreshes

    def fes_result(self, *, min_count=1):
        free, mask, diagnostics = integrate_mean_force(self.grid.edges, self.grid.mean,
                                                       counts=self.grid.count, min_count=min_count)
        return FESResult(self.grid.edges, free, np.full_like(free, np.nan), mask,
                         tuple(self.cv.units), "abf", "replica_spread",
                         metadata={"beta": self.beta, "measure": "cv_density", "converged": False,
                                   "physical_converged": False, "estimator": "projected_mean_force",
                                   "projection": diagnostics},
                         extras={"counts": self.grid.count.copy(), "raw_count": self.grid.count.copy(), "mean_force": self.grid.mean.copy(),
                                 "local_force_variance": self.grid.variance},
                         uncertainty_issue="one adaptive history; independent replicas or correlation-aware resampling required")


class ExtendedABF:
    """Auxiliary-coordinate eABF with exact scalar coupling derivatives.

    The auxiliary mass and temperature are physical (beta, not beta/P).
    ``coupling_bias`` is this object: pass it to the physical Dynamics step.
    Use ``advance`` to split auxiliary drift/kicks around that MD step.
    """

    def __init__(self, cv, edges, beta, position, *, coupling, mass, friction,
                 full_samples=200, wall_stiffness=1.0, momentum=None):
        self.bias = ABFBias(cv, edges, beta, full_samples=full_samples, wall_stiffness=wall_stiffness)
        self.cv, self.beta = self.bias.cv, self.bias.beta
        shape = (cv.dimension,)
        self.position = _array(position, "extended position", shape)
        self.momentum = np.zeros(shape) if momentum is None else _array(momentum, "extended momentum", shape)
        self.coupling = _array(np.broadcast_to(coupling, shape), "coupling")
        self.mass = _array(np.broadcast_to(mass, shape), "extended mass")
        self.friction = finite_positive(friction, "extended friction")
        if np.any(self.coupling <= 0) or np.any(self.mass <= 0):
            raise ValueError("extended coupling and mass must be positive")
        self.physical_grid = ForceGrid(edges)

    def energy_force(self, q):
        value, _, jac = _centroid_geometry(self.cv, q)
        delta = value - self.position
        return (len(q) * 0.5 * np.dot(self.coupling, delta * delta),
                -len(q) * np.einsum("d,dpnc->pnc", self.coupling * delta, jac))

    def _force(self, q):
        value = _centroid_geometry(self.cv, q)[0]
        return self.coupling * (value - self.position) - self.bias.physical_energy_gradient(self.position)[1]

    def advance(self, dynamics, state, physical_step):
        """Symmetric auxiliary OU/B/A, physical step, A/B/OU at fixed history.

        Caller owns transactional rollback of this state, the MD state and RNG
        if any substep fails. Adapt statistics only after a complete step.
        """
        dt = dynamics.dt
        if not np.isfinite(dt) or dt <= 0 or dynamics.beta != self.beta:
            raise ValueError("eABF requires positive dt and the same physical beta")
        decay = np.exp(-0.5 * self.friction * dt)
        noise = np.sqrt(self.mass / self.beta * (-np.expm1(-self.friction * dt)))
        self.momentum = decay * self.momentum + noise * dynamics.rng.normal(size=self.cv.dimension)
        self.momentum += 0.5 * dt * self._force(state.q)
        self.position += 0.5 * dt * self.momentum / self.mass
        physical_step(dynamics, state, self)
        self.position += 0.5 * dt * self.momentum / self.mass
        self.momentum += 0.5 * dt * self._force(state.q)
        self.momentum = decay * self.momentum + noise * dynamics.rng.normal(size=self.cv.dimension)
        if not np.isfinite(self.position).all() or not np.isfinite(self.momentum).all():
            raise FloatingPointError("nonfinite eABF auxiliary state")

    def observe(self, q):
        value = _centroid_geometry(self.cv, q)[0]
        restraint = self.coupling * (self.position - value)
        self.bias.observe(self.position, restraint)
        self.physical_grid.update(value, restraint)

    def state_dict(self):
        return {"schema": 1, "method": "eabf", "bias": self.bias.state_dict(),
                "coupling": self.coupling.tolist(), "mass": self.mass.tolist(), "friction": self.friction,
                "position": self.position.tolist(), "momentum": self.momentum.tolist(),
                "physical_grid": self.physical_grid.state_dict()}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError("invalid eABF checkpoint fields")
        for key in ("schema", "method", "coupling", "mass", "friction"):
            if state[key] != expected[key]:
                raise ValueError("eABF checkpoint identity differs: " + key)
        position = _array(state["position"], "extended position", self.position.shape)
        momentum = _array(state["momentum"], "extended momentum", self.position.shape)
        grid = ForceGrid(self.physical_grid.edges)
        grid.load_state_dict(state["physical_grid"])
        bias = ABFBias(self.cv, self.bias.grid.edges, self.beta, full_samples=self.bias.full_samples,
                       wall_stiffness=self.bias.wall_stiffness)
        bias.load_state_dict(state["bias"])
        if int(grid.count.sum()) + grid.outside != bias.observations:
            raise ValueError("eABF physical and extended observation clocks differ")
        self.bias = bias
        self.position, self.momentum, self.physical_grid = position, momentum, grid

    def fes_result(self, *, min_count=1):
        """CZAR: dA/dz = mean[k(lambda-z)|z] - d(log rho_tilde)/beta.

        Boundary bins and cells without a complete centered stencil are masked;
        no pseudocounts or interpolation across empty bins are introduced.
        """
        minimum = checked_integer(min_count, "min_count", 1)
        grid = self.physical_grid
        supported = grid.count >= minimum
        log_density = np.zeros(grid.shape)
        volume = np.ones(grid.shape)
        for axis, e in enumerate(grid.edges):
            volume *= np.diff(e).reshape(tuple(-1 if j == axis else 1 for j in range(grid.dimension)))
        log_density[supported] = np.log(grid.count[supported] / volume[supported])
        gradient = grid.mean.copy()
        valid = supported.copy()
        for axis, e in enumerate(grid.edges):
            if grid.shape[axis] < 3:
                valid[:] = False
                continue
            centers = (e[1:] + e[:-1]) * 0.5
            gradient[..., axis] -= np.gradient(log_density, centers, axis=axis, edge_order=2) / self.beta
            neighbors = np.roll(supported, 1, axis) & np.roll(supported, -1, axis)
            boundary = [slice(None)] * grid.dimension
            boundary[axis] = [0, -1]
            neighbors[tuple(boundary)] = False
            valid &= neighbors
        free, mask, diagnostics = integrate_mean_force(grid.edges, gradient,
                                                       counts=np.where(valid, grid.count, 0), min_count=minimum)
        return FESResult(grid.edges, free, np.full_like(free, np.nan), mask,
                         tuple(self.cv.units), "eabf", "replica_spread",
                         metadata={"beta": self.beta, "measure": "cv_density", "converged": False,
                                   "physical_converged": False, "estimator": "czar", "projection": diagnostics},
                         extras={"counts": grid.count.copy(), "raw_count": grid.count.copy(), "mean_force": gradient,
                                 "mean_restraint_force": grid.mean.copy(),
                                 "extended_counts": self.bias.grid.count.copy(),
                                 "extended_mean_force": self.bias.grid.mean.copy()},
                         uncertainty_issue="one adaptive history; CZAR histogram derivative has finite-bin and correlation error")
