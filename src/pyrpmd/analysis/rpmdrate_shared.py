"""One native RPMDrate coordinate for classical sampling and quantum operators.

No replacement xi formula is implemented here. The existing RPMDrateCoordinate
owns xi=s0/(s0-s1), equivalent paths, isotope masses and centroid semantics.
Quantum charts reject denominator sign changes and nonsmooth branch seams.
Atomic units; nonperiodic Cartesian input. No PMF-to-quantum-PES conversion.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import numpy as np
from scipy.linalg import null_space


def _real(value, name):
    x = np.asarray(value)
    if x.dtype.kind not in 'fiu' or not np.isfinite(x).all():
        raise ValueError(f'{name} must be finite real data')
    return np.array(x, dtype=float, copy=True)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return float(value)


def _json_default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError('Not a JSON scientific artifact: '+type(value).__name__)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False,default=_json_default).encode()).hexdigest()


def coordinate_definition(coordinate):
    """Serialize the native mathematical definition, not frame orientation."""
    if getattr(coordinate, 'estimator', None) != 'centroid':
        raise ValueError('The shared RPMDrate reaction coordinate must be a centroid estimator')
    masses = _real(coordinate.masses, 'masses')
    if masses.ndim != 1 or np.any(masses <= 0):
        raise ValueError('Positive native masses required')
    weights=np.zeros(len(masses))
    for sign,frag in zip((1.,-1.),coordinate.com.fragments):
        weights[list(frag)]=sign*masses[list(frag)]/masses[list(frag)].sum()
    if hasattr(coordinate.com,'weights') and not np.array_equal(coordinate.com.weights,weights):
        raise ValueError('Native COM weights disagree with fragment/isotope definition')
    return {'format': 'PyRPMD.shared-rpmdrate-coordinate.v1',
            'formula': 'xi=s0/(s0-s1)', 's0': 'rinf-COM_distance',
            's1': 'min_bond(max_equivalent_path(reference_forming-reference_breaking-r_forming+r_breaking))',
            'estimator': 'xi(mean_bead_positions)', 'index_base': 0,
            'length_unit': 'bohr', 'mass_unit': 'electron_mass',
            'masses': masses.tolist(), 'com_weights':weights.tolist(), 'fragments': [list(g) for g in coordinate.com.fragments],
            'rinf': _positive(coordinate.rinf, 'rinf'),
            'reference_difference': _real(coordinate.reference_difference, 'reference differences').tolist(),
            'paths': [[[list(pair) for pair in forming], [list(pair) for pair in breaking]]
                      for forming, breaking in coordinate.paths]}


@dataclass(frozen=True)
class SharedPoint:
    xi: float
    surface: float
    gradient: np.ndarray
    surface_gradient: np.ndarray
    denominator: float
    branch: tuple[int, int]
    branch_margin_bohr: float


class SharedRPMDrateCoordinate:
    """Read-only-definition adapter around the *same* native coordinate instance.

    Native min/max selection is reproduced ONLY for seam diagnostics, not for
    evaluation of xi/its derivatives. The native coordinate may be used by
    ordinary PMF/parent/child workflows without this stricter quantum seam test.
    """
    def __init__(self, coordinate, xi_star, *, seam_tolerance=1e-8, denominator_floor=1e-8):
        self.coordinate = coordinate
        value=_real(xi_star,'xi_star')
        if value.ndim!=0:raise ValueError('xi_star must be a scalar')
        self.xi_star = float(value)
        self.seam_tolerance = _positive(seam_tolerance, 'seam_tolerance')
        self.denominator_floor = _positive(denominator_floor, 'denominator_floor')
        self._definition = coordinate_definition(coordinate)
        self.coordinate_id = _digest(self._definition)
        raw=np.asarray(self._definition['masses'],dtype=float)
        self.masses = np.frombuffer(raw.tobytes(),dtype=float)
        self.masses.flags.writeable = False
        self.surface_id = _digest({'coordinate_id': self.coordinate_id, 'xi_star': self.xi_star,
                                  'surface': '(1-xi_star)*s0+xi_star*s1',
                                  'product_side': 'positive native surface',
                                  'quantum_domain': 's0-s1>0'})
        self._native_surface = coordinate.dividing_surface(self.xi_star)

    @classmethod
    def from_rate_job(cls, config, system, xi_star, **kwargs):
        """Reuse PyRPMD's public native coordinate factory. No atom reordering."""
        if config.get('task', 'rate') != 'rate' or any(getattr(system, 'pbc', ())) or getattr(system, 'fixed_atoms', ()):
            raise ValueError('Only nonperiodic, unconstrained gas-phase rate jobs are supported')
        from ..workflows.rate_constraints import rate_coordinate
        return cls(rate_coordinate(config, system), xi_star, **kwargs)

    def definition(self):
        self._check_unchanged()
        return json.loads(json.dumps(self._definition))

    def _check_unchanged(self):
        if _digest(coordinate_definition(self.coordinate)) != self.coordinate_id:
            raise ValueError('Native coordinate changed after binding; rebuild all shared artifacts')
        if float(self._native_surface.xi) != self.xi_star:
            raise ValueError('Native dividing surface changed')

    def _geometry(self, positions):
        q = _real(positions, 'positions')
        if q.shape != (len(self.masses), 3):
            raise ValueError('A physical geometry must have shape [atoms,3]')
        return q

    def _branch(self, q):
        terms = []
        for forming, breaking in self.coordinate.paths:
            terms.append([ref - np.linalg.norm(q[f[0]]-q[f[1]]) + np.linalg.norm(q[b[0]]-q[b[1]])
                          for ref, f, b in zip(self.coordinate.reference_difference, forming, breaking)])
        terms = np.asarray(terms, float)
        columns = terms.max(axis=0)
        bond = int(columns.argmin()); path = int(terms[:, bond].argmax())
        margins = []
        if terms.shape[0] > 1:
            top = np.sort(terms[:, bond]); margins.append(top[-1] - top[-2])
        if terms.shape[1] > 1:
            low = np.sort(columns); margins.append(low[1] - low[0])
        return (path, bond), min(margins) if margins else np.inf

    def evaluate(self, positions, *, require_smooth=False):
        self._check_unchanged(); q = self._geometry(positions)
        s0, _, _, s1, _, _ = self.coordinate.surfaces(q)
        denominator = float(s0-s1)
        if not np.isfinite(denominator) or denominator <= self.denominator_floor:
            raise ValueError('Quantum chart requires positive nonsingular s0-s1; native xi must not be clipped')
        branch, margin = self._branch(q)
        if require_smooth and margin <= self.seam_tolerance:
            raise ValueError('Equivalent-path/bond seam is nondifferentiable; use separate patches')
        xi, g = self.coordinate.centroid_value_gradient(q)
        surface, gs = self._native_surface.centroid_value_gradient(q)
        g = _real(g, 'native gradient'); gs = _real(gs, 'native surface gradient')
        if g.shape != q.shape or gs.shape != q.shape or not np.isfinite([xi, surface]).all():
            raise ValueError('Native coordinate returned invalid values')
        if not np.isclose(surface, denominator*(xi-self.xi_star), rtol=2e-10, atol=2e-10):
            raise ValueError('Native xi and dividing surface disagree')
        g.flags.writeable = False; gs.flags.writeable = False
        return SharedPoint(float(xi), float(surface), g, gs, denominator, branch, float(margin))

    def ring_value_gradient(self, beads):
        q = _real(beads, 'beads')
        if q.ndim != 3 or q.shape[0] < 1 or q.shape[1:] != (len(self.masses), 3):
            raise ValueError('Need [beads,atoms,3]')
        point = self.evaluate(q.mean(axis=0))
        return point.xi, np.broadcast_to(point.gradient / len(q), q.shape).copy()

    def side(self, positions):
        return float(self.evaluate(positions).surface > 0)

    def centroid_flux_velocity(self, beads, momenta):
        q = _real(beads, 'beads'); p = _real(momenta, 'momenta')
        self.ring_value_gradient(q)
        if p.shape != q.shape: raise ValueError('Momentums must match bead coordinates')
        g = self.evaluate(q.mean(axis=0)).surface_gradient
        return float(np.sum(g * p.mean(axis=0) / self.masses[:, None]))

    def physical_mass_metric(self, positions, auxiliary_gradients=()):
        """Contravariant G_ab in physical coordinate space (not fictitious bead mass).

        This projected block is a diagnostic, NOT the inverse of a frozen-bath
        induced kinetic metric. A complete chart or explicit reduction is needed.
        """
        g = self.evaluate(positions, require_smooth=True).gradient.ravel()
        rows = [g]
        for a in auxiliary_gradients:
            a = _real(a, 'auxiliary gradient')
            if a.shape != (len(self.masses), 3): raise ValueError('Wrong auxiliary gradient')
            rows.append(a.ravel())
        J = np.asarray(rows); G = (J / np.repeat(self.masses, 3)) @ J.T
        if np.linalg.eigvalsh(G).min() <= 1e-12*np.linalg.eigvalsh(G).max():
            raise ValueError('Dependent or singular coordinate gradients')
        return G


class LocalXiChart:
    """Full local Cartesian chart u=(native xi, orthogonal mass-weighted coordinates).

    Invert by moving along a reference normal, not by assuming xi is linear.
    The full Jacobian determines kinetic cross terms. For an active slice, the
    metric is explicitly the *constrained-manifold* metric, not an environment
    influence functional. No zero-point/bath term is invented.
    """
    def __init__(self, shared, reference, *, auxiliary_directions=None, radius=1.0):
        self.shared = shared; self.reference = shared._geometry(reference)
        point = shared.evaluate(self.reference, require_smooth=True)
        self.branch = point.branch; self.xi_reference = point.xi
        self.sqrt_mass = np.repeat(np.sqrt(shared.masses), 3)
        normal = point.gradient.ravel()/self.sqrt_mass
        normal /= np.linalg.norm(normal)
        basis = [normal]
        if auxiliary_directions is not None:
            for v in _real(auxiliary_directions, 'mass-weighted directions'):
                if v.shape != normal.shape: raise ValueError('Auxiliary direction has wrong size')
                v = v.copy()
                for old in basis: v -= old * (old @ v)
                if np.linalg.norm(v) < 1e-10: raise ValueError('Dependent auxiliary direction')
                basis.append(v/np.linalg.norm(v))
        Q = np.column_stack(basis)
        self.basis = np.column_stack([Q, null_space(Q.T)])
        self.radius = _positive(radius, 'chart radius')
        self.dimension = len(normal)

    def _point(self, lam, bath):
        y = np.r_[lam, bath]
        if np.linalg.norm(y) > self.radius: raise ValueError('Outside local chart trust region')
        x = self.reference + (self.basis@y/self.sqrt_mass).reshape(self.reference.shape)
        point = self.shared.evaluate(x, require_smooth=True)
        if point.branch != self.branch: raise ValueError('Chart crossed a native coordinate branch')
        return x, point

    def forward(self, u):
        u = _real(u, 'chart point')
        if u.shape != (self.dimension,): raise ValueError('Provide all active and environment coordinates')
        target = u[0]; bath = u[1:]; lam = 0.0
        direction = self.basis[:, 0]/self.sqrt_mass
        for _ in range(50):
            x, point = self._point(lam, bath)
            err = point.xi-target
            if abs(err) < 2e-12: return x
            derivative = point.gradient.ravel()@direction
            if abs(derivative) < 1e-12: raise ValueError('Reference-normal chart became singular')
            step = -err/derivative
            accepted = False
            for scale in (.5**np.arange(20)):
                candidate = lam+scale*step
                try: _, trial = self._point(candidate, bath)
                except ValueError: continue
                if abs(trial.xi-target) < abs(err): lam=candidate; accepted=True; break
            if not accepted: raise ValueError('No smooth local inverse for requested xi')
        raise ValueError('Xi inversion did not converge')

    def inverse(self, positions):
        x = self.shared._geometry(positions)
        y = self.basis.T@((x-self.reference).ravel()*self.sqrt_mass)
        checked, point = self._point(y[0], y[1:])
        if not np.allclose(x, checked, atol=1e-11, rtol=1e-11): raise ValueError('Coordinate map mismatch')
        return np.r_[point.xi, y[1:]]

    def jacobian(self, u):
        x = self.forward(u); point = self.shared.evaluate(x, require_smooth=True)
        L = self.basis/self.sqrt_mass[:, None];g = point.gradient.ravel();d = g@L[:, 0]
        if abs(d) < 1e-12: raise ValueError('Singular xi Jacobian')
        J = L.copy();J[:, 0] /= d;J[:, 1:] -= L[:, 0, None]*(g@L[:, 1:])[None, :]/d
        return J

    def geometry(self, u):
        J = self.jacobian(u);covariant = J.T@(self.sqrt_mass[:, None]**2*J)
        return np.linalg.inv(covariant), float(abs(np.linalg.det(J)))

    def active_slice_metric(self, u, dimension):
        if type(dimension) is not int or not 1 <= dimension <= min(3, self.dimension):
            raise ValueError('Active dimension must be 1, 2 or 3')
        J = self.jacobian(u)[:, :dimension];g=J.T@(self.sqrt_mass[:, None]**2*J)
        return {'inverse_mass':np.linalg.inv(g),'density':float(np.sqrt(np.linalg.det(g))),
                'reduction':'constrained active manifold; frozen omitted chart coordinates',
                'bath_feedback_included':False,'physical_converged':False}
