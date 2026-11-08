"""Share the native RPMDrate coordinate with an explicit quantum reference.

xi=s0/(s0-s1); S=(1-xi_star)*s0+xi_star*s1. The authoritative product
indicator is S>0, not xi>xi_star on the wrong side of a ratio pole. This
adapter delegates geometry to the existing CV and never replaces its min/max
rules. A restricted path quantum calculation is NOT a reduced bath theory.
"""
from __future__ import annotations
import copy
import hashlib
import json
import numpy as np
from scipy.optimize import brentq
from ..io import jsonable


def _real(x, name):
    a = np.asarray(x)
    if a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite real data')
    return np.asarray(a, float)


def _digest(record):
    return hashlib.sha256(json.dumps(jsonable(record), sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


class SharedRateCoordinate:
    """Snapshot a native RPMDrateCoordinate; detect later mutation of that copy."""
    def __init__(self, coordinate, *, beta, xi_star, pes_identity, normalization_id):
        from ..cv import RPMDrateCoordinate
        if type(coordinate) is not RPMDrateCoordinate:
            raise TypeError('An unmodified native RPMDrateCoordinate is required')
        if (isinstance(beta,(bool,np.bool_)) or isinstance(xi_star,(bool,np.bool_))
                or not np.isfinite([beta, xi_star]).all() or beta <= 0
                or not isinstance(pes_identity, str) or not pes_identity
                or not isinstance(normalization_id, str) or not normalization_id):
            raise ValueError('Positive beta, finite surface and explicit identities required')
        self._coordinate = copy.deepcopy(coordinate)
        self.beta = float(beta)
        self.xi_star = float(xi_star)
        self.pes_identity = pes_identity
        self.normalization_id = normalization_id
        self._identity = _digest(self._record())

    @classmethod
    def from_job(cls, config, system, *, xi_star, normalization_id):
        from ..workflows.rate_constraints import rate_coordinate
        if config.get('task', 'rate') != 'rate' or any(system.pbc) or system.fixed_atoms:
            raise ValueError('This bridge supports gas-phase rate jobs, not surface/periodic jobs')
        return cls(rate_coordinate(config, system), beta=config['beta'], xi_star=xi_star,
                   pes_identity=_digest(config['potential']), normalization_id=normalization_id)

    def _record(self):
        c = self._coordinate
        return {'schema': 'PyRPMD.shared-rate-coordinate.v1', 'beta': self.beta,
                'xi_star': self.xi_star, 'pes_identity': self.pes_identity,
                'normalization_id': self.normalization_id, 'fragments': c.com.fragments,
                'masses': c.masses.tolist(), 'com_weights':c.com.weights.tolist(), 'rinf': c.rinf,
                'reference_difference': list(c.reference_difference), 'paths': c.paths,
                'coordinate': 's0/(s0-s1)', 'side': '((1-xi_star)*s0+xi_star*s1)>0',
                'rp_estimator': 'xi(centroid Cartesian positions)',
                'quantum_lift': 'same physical-coordinate surface, not a centroid operator identity'}

    @property
    def identity(self):
        if _digest(self._record()) != self._identity:
            raise ValueError('Shared coordinate/temperature/normalization was mutated')
        return self._identity

    def manifest(self):
        return {**self._record(), 'identity': self.identity}

    @property
    def masses(self):
        self.identity
        return self._coordinate.masses.copy()

    def _point(self, q):
        self.identity
        q = _real(q, 'Cartesian point')
        if q.shape != (len(self.masses), 3):
            raise ValueError('Expected physical [atoms,3] positions in bohr')
        return q

    def side(self, q):
        q = self._point(q)
        value, _ = self._coordinate.dividing_surface(self.xi_star).centroid_value_gradient(q)
        return float(value > 0)

    def geometry(self, q, *, require_smooth=True):
        q = self._point(q)
        c = self._coordinate
        s0, g0, _, s1, g1, _ = c.surfaces(q)
        den = s0 - s1
        if den <= 1e-8:
            raise ValueError('Quantum xi chart requires a positive, nonsingular s0-s1 patch')
        if require_smooth:
            terms = np.array([[ref - np.linalg.norm(q[f[0]]-q[f[1]])
                               + np.linalg.norm(q[b[0]]-q[b[1]])
                               for ref, f, b in zip(c.reference_difference, fs, bs)]
                              for fs, bs in c.paths])
            col = terms.max(axis=0)
            chosen = int(np.argmin(col))
            if (np.count_nonzero(np.abs(col-col[chosen]) < 1e-9) > 1
                    or np.count_nonzero(np.abs(terms[:, chosen]-col[chosen]) < 1e-9) > 1):
                raise ValueError('Min/max branch seam: no smooth quantum coordinate chart')
        xi, g = c.centroid_value_gradient(q)
        S = (1-self.xi_star)*s0 + self.xi_star*s1
        gS = (1-self.xi_star)*g0 + self.xi_star*g1
        return {'xi': float(xi), 'surface': float(S), 'xi_gradient': g,
                'surface_gradient': gS,
                'xi_inverse_mass': float(np.sum(g*g/self.masses[:, None])),
                'surface_positive_flux_speed': float(np.sqrt(np.sum(gS*gS/self.masses[:, None])/(2*np.pi*self.beta)))}

    def centroid_geometry(self, beads):
        q = _real(beads, 'beads')
        if q.ndim != 3 or q.shape[1:] != (len(self.masses), 3) or len(q) < 1:
            raise ValueError('Expected [beads,atoms,3], not RPMDrate Fortran array order')
        return self.geometry(q.mean(axis=0))

    def active_metric(self, q, auxiliaries=()):
        """Contravariant CV block; NOT a complete reduced kinetic operator."""
        if len(auxiliaries) > 2:
            raise ValueError('At most xi plus two auxiliary coordinates supported')
        data = self.geometry(q)
        jac = [data['xi_gradient']]
        values = [data['xi']]
        for cv in auxiliaries:
            value, grad = cv.centroid_value_gradient(q)
            grad = _real(grad, 'auxiliary gradient')
            if grad.shape != jac[0].shape or not np.isfinite(value):
                raise ValueError('Invalid auxiliary coordinate')
            values.append(float(value)); jac.append(grad)
        jac = np.array(jac)
        G = np.einsum('aij,bij,i->ab', jac, jac, 1/self.masses)
        if np.linalg.eigvalsh(G).min() <= 1e-12*np.linalg.eigvalsh(G).max():
            raise ValueError('Dependent/singular selected coordinates')
        return {'values': np.array(values), 'jacobian': jac, 'inverse_mass_block': G,
                'full_kinetic_metric_available': False, 'identity': self.identity}


class XiSlice:
    """Explicit fixed-transverse path q=anchor+tangent*z parameterized by xi.

    This is a restricted 1D Hamiltonian, not PMF-as-potential and not a claim
    to have integrated out the omitted quantum/environment degrees of freedom.
    """
    def __init__(self, shared, anchor, tangent, bracket):
        self.shared = shared
        self.anchor = shared._point(anchor).copy()
        self.tangent = _real(tangent, 'tangent').copy()
        self.bracket = tuple(_real(bracket, 'bracket'))
        if (self.tangent.shape != self.anchor.shape or np.linalg.norm(self.tangent) == 0
                or len(self.bracket) != 2 or self.bracket[0] >= self.bracket[1]):
            raise ValueError('Invalid path tangent or bracket')
        self.identity = shared.identity
        derivative = []
        for z in np.linspace(*self.bracket, 129):
            d = shared.geometry(self.anchor+z*self.tangent)
            derivative.append(np.sum(d['xi_gradient']*self.tangent))
        derivative = np.array(derivative)
        if np.any(np.abs(derivative) < 1e-10) or not (np.all(derivative>0) or np.all(derivative<0)):
            raise ValueError('Path is not a nonsingular monotone xi chart')
        self.path_mass = float(np.sum(shared.masses[:, None]*self.tangent**2))
        self._path_digest=_digest([self.anchor.tolist(),self.tangent.tolist(),self.bracket,self.path_mass])

    def map(self, xi):
        if (_digest([self.anchor.tolist(),self.tangent.tolist(),self.bracket,self.path_mass])!=self._path_digest
                or self.shared.identity != self.identity):
            raise ValueError('Coordinate identity changed')
        values = _real(xi, 'xi grid')
        positions = []; mass = []
        for target in values.ravel():
            def residual(z):
                return self.shared.geometry(self.anchor+z*self.tangent)['xi']-target
            z = brentq(residual, *self.bracket, xtol=1e-13)
            q = self.anchor+z*self.tangent
            derivative = np.sum(self.shared.geometry(q)['xi_gradient']*self.tangent)
            positions.append(q); mass.append(self.path_mass/derivative**2)
        return np.array(positions), np.array(mass)

    def geometry(self, points):
        if np.shape(points)[-1] != 1:
            raise ValueError('XiSlice has exactly one propagation coordinate')
        _, mass = self.map(np.asarray(points)[:, 0])
        return (1/mass)[:, None, None], np.sqrt(mass)

    def potential(self, points, pes):
        q, _ = self.map(np.asarray(points)[:, 0])
        result = pes.evaluate(q)
        if hasattr(result,'status') and np.any(result.status):
            raise ValueError('Physical PES reported failed configurations')
        value = _real(result.energy, 'physical PES energies')
        if value.shape != (len(q),):
            raise ValueError('PES must evaluate physical configurations, not a PMF')
        return value


class XiAffineChart:
    """1--3D affine Cartesian core, reparameterized as (xi,aux1,aux2).

    The Cartesian anchor and directions define a restricted physical PES, with
    omitted directions fixed. Auxiliary coordinates are the explicit affine
    coefficients, NOT an undocumented bath average. The induced metric includes
    mixed kinetic terms. Validity/monotonicity is checked locally on each call;
    a finite probe grid does not prove a globally smooth coordinate atlas.
    """
    def __init__(self, shared, anchor, directions, bracket):
        self.shared=shared;self.anchor=shared._point(anchor).copy()
        self.directions=_real(directions,'directions').copy()
        self.bracket=tuple(_real(bracket,'bracket'))
        if (self.directions.ndim!=3 or self.directions.shape[1:]!=self.anchor.shape
                or not 1<=len(self.directions)<=3 or len(self.bracket)!=2
                or self.bracket[0]>=self.bracket[1]):
            raise ValueError('Explicit 1--3 Cartesian core directions and bracket required')
        mass=np.einsum('aij,bij,i->ab',self.directions,self.directions,shared.masses)
        ev=np.linalg.eigvalsh(mass)
        if ev.min()<=1e-12*ev.max():raise ValueError('Dependent affine core directions')
        self.identity=shared.identity
        self._digest=_digest([self.anchor,self.directions,self.bracket])

    def _mapped(self, points):
        u=_real(points,'core grid')
        if u.ndim!=2 or u.shape[1]!=len(self.directions) or not len(u):
            raise ValueError('Expected [points,core_dimension]')
        if (self.identity!=self.shared.identity
                or _digest([self.anchor,self.directions,self.bracket])!=self._digest):
            raise ValueError('Shared affine quantum chart was mutated')
        positions=[];jacobians=[]
        for value in u:
            base=self.anchor+np.einsum('a,aij->ij',value[1:],self.directions[1:])
            tangent=self.directions[0]
            def residual(z):return self.shared.geometry(base+z*tangent)['xi']-value[0]
            signs=[]
            for z in np.linspace(*self.bracket,9):
                g=self.shared.geometry(base+z*tangent)['xi_gradient']
                signs.append(float(np.sum(g*tangent)))
            if min(abs(x) for x in signs)<1e-10 or min(signs)*max(signs)<=0:
                raise ValueError('Affine core is not a monotone smooth xi chart on this patch')
            z=brentq(residual,*self.bracket,xtol=1e-13)
            q=base+z*tangent;g=self.shared.geometry(q)['xi_gradient']
            derivatives=np.einsum('ij,aij->a',g,self.directions)
            if abs(derivatives[0])<1e-10:raise ValueError('Singular xi chart')
            jac=self.directions.copy();jac[0]=tangent/derivatives[0]
            for j in range(1,len(jac)):jac[j]-=tangent*derivatives[j]/derivatives[0]
            positions.append(q);jacobians.append(jac)
        return np.asarray(positions),np.asarray(jacobians)

    def map(self,points):
        return self._mapped(points)[0]

    def geometry(self,points):
        _,J=self._mapped(points)
        g=np.einsum('naij,nbij,i->nab',J,J,self.shared.masses)
        return np.linalg.inv(g),np.sqrt(np.linalg.det(g))

    def potential(self,points,pes):
        q=self.map(points);result=pes.evaluate(q)
        if hasattr(result,'status') and np.any(result.status):raise ValueError('Physical PES failed')
        value=_real(result.energy,'physical PES energies')
        if value.shape!=(len(q),):raise ValueError('PES energy shape mismatch')
        return value
