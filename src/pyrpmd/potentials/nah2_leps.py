"""The published NaH2 5G two-state molecular potential, in native atomic units.

NumPy translation of the POTLIB/ChemPotPy NaH2_LEPS_5G_1992_DPEM.f kernel.
Original fit: P. Halvick and D. G. Truhlar, JCP 96, 2895 (1992),
doi:10.1063/1.461986; erratum JCP 100, 4718 (1994), doi:10.1063/1.467282.
Source: shuyinan/chempotpy@4716d22e87ae01770cad9ec995c4a02a8e479296.
Distributed upstream under Apache-2.0; see _licenses/chempotpy-APACHE-2.0.txt.

Changes from that source: vectorized Python evaluation; exact chain-rule
forward differentiation of its energy expressions; atom-order restoration;
exceptions instead of Fortran STOP. All fit constants, including historical
eV-to-hartree factors and the common 5G energy shift, are preserved. This port
evaluates the native POT kernel, not the rounded angstrom/eV DPEM wrapper.
"""

from __future__ import annotations

import numpy as np

from .base import Potential, PotentialError, coordinates


_SOURCE = {
    "repository": "https://github.com/shuyinan/chempotpy",
    "commit": "4716d22e87ae01770cad9ec995c4a02a8e479296",
    "path": "chempotpy/NaH2/NaH2_LEPS_5G_1992_DPEM.f",
    "sha256": "b98e2cb6a7355d868346b2a901441998705f9dead1dd233146285fd618d08f61",
    "license": "Apache-2.0",
    "native_units": "hartree,bohr",
    "translation": "numpy_forward_derivative_v1",
}
_CITATIONS = ("10.1063/1.461986", "10.1063/1.467282", "10.1021/acs.jpca.3c05899")


class _Jet:
    """A batch of values and derivatives with respect to three bond distances."""
    __array_priority__ = 1000

    def __init__(self, value, gradient):
        self.v, self.g = np.asarray(value), np.asarray(gradient)

    def __add__(self, other):
        if isinstance(other, _Jet):
            return _Jet(self.v+other.v, self.g+other.g)
        return _Jet(self.v+other, self.g)

    __radd__ = __add__

    def __neg__(self):
        return _Jet(-self.v, -self.g)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if isinstance(other, _Jet):
            return _Jet(self.v*other.v, self.g*other.v[:, None]+other.g*self.v[:, None])
        return _Jet(self.v*other, self.g*other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, _Jet):
            value = self.v/other.v
            return _Jet(value, (self.g-value[:, None]*other.g)/other.v[:, None])
        return _Jet(self.v/other, self.g/other)

    def __rtruediv__(self, other):
        return self**(-1) * other

    def __pow__(self, power):
        return _Jet(self.v**power, (power*self.v**(power-1))[:, None]*self.g)

    def __abs__(self):
        return _Jet(np.abs(self.v), np.sign(self.v)[:, None]*self.g)


def _exp(value):
    result = np.exp(value.v)
    return _Jet(result, result[:, None]*value.g)


def _tanh(value):
    result = np.tanh(value.v)
    return _Jet(result, (1-result*result)[:, None]*value.g)


def _h2_singlet(r):
    return ((149.480-59.6557*r)*_exp(-4.13792*r)
            + r*r*(-23.7299+3.91747*r)*_exp(-1.41350*r))


def _leps(s1, s2, s3, t1, t2, t3, total_r, c2a, c2b):
    coul1, coul2, coul3 = .5*(s1+t1), .5*(s2+t2), .5*(s3+t3)
    exch1, exch2, exch3 = .5*(s1-t1), .5*(s2-t2), .5*(s3-t3)
    w = (exch1-exch2)**2+(exch2-exch3)**2+(exch3-exch1)**2
    coupling = c2a*_exp(-c2b*w-.15*total_r)
    return coul1+coul2+coul3+4.7469-(w+coupling*coupling)**.5/np.sqrt(2.)


def _lower(r1, r2, r3, cosine_squared):
    c1 = 6.279847+(10.60311-6.279847)*cosine_squared
    c2 = .003849003+(.006058605-.003849003)*cosine_squared
    c3 = 5.546257+(4.573378-5.546257)*cosine_squared
    c4 = .5573967+(.5182708-.5573967)*cosine_squared
    switch = (.5*(1-_tanh(.7*(r1-6.5)))*.5*(1-_tanh(.7*(r3-6.5)))
              *.5*(1+_tanh(1.1*(r2-1.9))))

    def singlet(r):
        ax = _exp(-.7536929*(r-5.015440))
        a = .08*ax*(ax-2.)
        cx = _exp(-.54070*(r-4.))
        c = 1.323042*cx*(cx-2.)
        return a+(c-a)*switch

    t1 = c1*_exp(-c2*r1**6)+c3*_exp(-c4*r1)
    t3 = c1*_exp(-c2*r3**6)+c3*_exp(-c4*r3)
    xt2 = _exp(-10.38036*(r2-1.237624))
    t2 = .02885656*xt2*(xt2-2.)
    return (_leps(singlet(r1), _h2_singlet(r2), singlet(r3), t1, t2, t3,
                  r1+r2+r3, 1.5, 1.) / 27.2113961 - 1.8427829108061610e-4)


def _coupling(r1, r2, r3):
    amp = .235737419+(1.51495-.235737419)*.5*(_tanh(.502447648*(r2-4.05991868))+1)
    erl = 230.065420+(.071-230.065420)*.5*(_tanh(3.46985415*(r2-.895373168))+1)
    ers = 2.+(11.683-2.)*.5*(_tanh(45.4595148*(r2+2.38192975))+1)
    rmx = 2.33484458+(2.62222-2.33484458)*.5*(_tanh(20.4523701*(r2-1.98941949))+1)
    f1 = amp*_exp(-(erl+ers/(1e-4+r1**3))*(r1-rmx)**2)
    f3 = amp*_exp(-(erl+ers/(1e-4+r3**3))*(r3-rmx)**2)
    r16, r36 = r1**6, r3**6
    ff = ((r1-r3)/r2)**2
    fg = ((r16-r36)/(r16+r36))**2
    hr = (r1*r1+r2*r2+r3*r3)**.5
    fh = .5*(1+_tanh(.75*(11-hr)))
    # This older conversion factor is explicitly retained in PREPOT12.
    return ((ff-fg)*fh+fg)*(f1+f3)/27.211611


def _upper(r1, r2, r3, rnag, cosine_squared):
    swa = .872116+(.614393-.872116)*cosine_squared
    swr0 = 8.194467+(9.006323-8.194467)*cosine_squared

    def singlet(r):
        return ((261.4452-.0003155166*r)*_exp(-1.785467*r)
                + r*r*(-3.548074+.3356766*r)*_exp(-.7621511*r))

    x90 = _exp(-.9514887*(r2-2.002516))
    t902a = 1.106187*x90*(x90-2.)
    x0 = _exp(-.8497374*(r2-2.300265))
    t02a = .8870172*x0*(x0-2.)
    t2a = t902a+(t02a-t902a)*cosine_squared
    switch = .5*(1-_tanh(swa*(.5*(r1+r3)-swr0)))
    s2a = _h2_singlet(r2)+2.1037
    s2b = (144.893*_exp(-3.85716*r2)
           + r2*(37.5919-4.32985*r2-.003807*r2*r2)*_exp(-1.52496*r2))
    t2 = switch*(t2a-s2b)+s2b
    s2 = .5*(s2a+t2-_tanh((s2a-t2)/.1)*(s2a-t2))

    def triplet(r):
        x90 = _exp(-.5127734*(r-7.233552))
        t90 = .01372065*x90*(x90+2.)
        x0 = _exp(-.6960127*(r-3.708055))
        t0 = .2417800*x0*(x0+2.)
        return t90+(t0-t90)*cosine_squared

    energy = _leps(singlet(r1), s2, singlet(r3), triplet(r1), t2, triplet(r3),
                   r1+r2+r3, .8, .8)
    rf11 = 2.59329096*(1+_tanh(9.93116245*(r2+6.85093745)))
    rf12 = .100540010*(1+_tanh(-3.09412207*(r2-1.74921175)))
    rf13 = .0680366735*(1+_tanh(4.47330630*(r2-1.70178393)))
    tr0 = rf11+cosine_squared*(rf12+rf13*cosine_squared)
    sine_squared = 1-cosine_squared
    rf21 = 4.52729085*(1+_tanh(-.715957152*(r2+.534701010)))
    rf22 = .0941855078*(1+_tanh(-7.86831467*(r2-1.90926538)))
    rf23 = .0284176599*(1+_tanh(-6.59302513*(r2-6.09626510)))
    rat = rf21+sine_squared*(rf22+rf23*sine_squared)
    al1 = 9+_exp(2.38029900*(1.47400334-r2))*(
        -.226331771-2.17339562*r2+9.54597998*r2**2-23.3366749*r2**3
        +24.7115748*r2**4-12.8786102*r2**5+2.75711758*r2**6)
    al2 = 9+_exp(1.35164094*(-.254404564-r2))*(
        -10.6627055-14.0112406*r2-7.80660398*r2**2+5.47697753*r2**3
        -6.95875664*r2**4+3.64334771*r2**5-.492505596*r2**6)
    alt = (2./3.)*(al1-al2)
    alb = (1./3.)*al1+(2./3.)*al2
    alhh = alb+alt*(1.5*cosine_squared-.5)
    c6 = 505.5*alhh/(18.35755920+(alb*.5)**.5)
    dispersion = -c6*rnag**(-6)*(1-_exp((-rnag/tr0)**7))
    cutoff = .5*(1+_tanh(rnag-r2))
    return energy/27.2113961+dispersion*cutoff*rat-1.8427829108061610e-4


def _native_radial(distances):
    basis = np.broadcast_to(np.eye(3), (len(distances), 3, 3))
    r1, r2, r3 = (_Jet(distances[:, k], basis[:, k]) for k in range(3))
    rnag = (.5*abs(r1*r1+r3*r3-.5*r2*r2))**.5
    if np.any(rnag.v < 1e-10):
        raise PotentialError("NaH2 5G is singular when Na is at the H2 midpoint (RNAG < 1e-10 bohr)")
    cosine_squared = (.5*(r1*r1-r3*r3)/(r2*rnag))**2
    return _lower(r1, r2, r3, cosine_squared), _coupling(r1, r2, r3), _upper(
        r1, r2, r3, rnag, cosine_squared)


class NaH2LEPS5G(Potential):
    """Full-dimensional 2x2 NaH2 5G diabatic matrix and Cartesian gradients.

    ``symbols`` fixes the caller's atom order; it must contain one Na and two H.
    Hydrogen labels stay in that order for every call; there is no geometry-
    dependent sorting or electronic phase matching. Matrix elements follow the
    original global diabatic convention. ``q[B,3,3]`` is in bohr. Outputs are
    hartree and positive hartree/bohr gradients, including off-diagonal terms.

    This historical fitted two-state surface describes Na(3p)+H2 quenching
    and reaction to NaH+H. It omits spin-orbit and higher electronic states.
    Mathematical domain checks are not a certification of the fit's accuracy
    for arbitrary high-energy or short-range configurations.
    """
    natoms = 3
    nstates = 2
    ncoordinates = 9
    coordinates = tuple(range(9))

    def __init__(self, *, symbols=("Na", "H", "H")):
        if (not isinstance(symbols, (list, tuple)) or len(symbols) != 3
                or any(not isinstance(s, str) for s in symbols)
                or sorted(symbols) != ["H", "H", "Na"]):
            raise ValueError("NaH2 5G symbols must contain one Na and two H in the supplied atom order")
        self.symbols = tuple(symbols)
        self._order = (self.symbols.index("Na"), *(i for i, s in enumerate(symbols) if s == "H"))
        self._frozen = True

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("molecular potential parameters are immutable")
        object.__setattr__(self, name, value)

    def to_dict(self):
        return {"schema": "pyrpmd.molecular_potential", "version": 1,
                "model": "nah2_leps_5g_1992", "parameters": {"symbols": list(self.symbols)},
                "source": dict(_SOURCE)}

    @property
    def metadata(self):
        return {"provider": "molecular_diabatic", "system": "NaH2", "nstates": 2, "natoms": 3,
                "charge": 0, "multiplicity": 2,
                "symbols": list(self.symbols), "execution_device": "cpu", "thread_safety": "reentrant",
                "trajectory_batch_safe": True, "periodic": False, "derivative": "gradient",
                "energy_unit": "hartree", "length_unit": "bohr", "electronic_basis": "fixed_diabatic",
                "citations": list(_CITATIONS), "model_identity": self.to_dict()}

    def evaluate(self, q):
        from ..nonadiabatic import DiabaticResult, validate_diabatic_result

        q = coordinates(q)
        if q.shape[1] != 3:
            raise PotentialError("NaH2 5G requires exactly three atoms")
        ordered = q[:, self._order, :]
        pairs = ((0, 1), (1, 2), (0, 2))
        vectors = np.stack([ordered[:, a]-ordered[:, b] for a, b in pairs], axis=1)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            distances = np.linalg.norm(vectors, axis=-1)
            if not np.isfinite(distances).all() or np.any(distances < 1e-10):
                raise PotentialError("NaH2 5G requires finite, noncoincident nuclei (pair distances >= 1e-10 bohr)")
            lower, coupling, upper = _native_radial(distances)
            matrix = np.empty((len(q), 2, 2))
            matrix[:, 0, 0], matrix[:, 1, 1] = lower.v, upper.v
            matrix[:, 0, 1] = matrix[:, 1, 0] = coupling.v
            radial = np.empty((len(q), 2, 2, 3))
            radial[:, 0, 0], radial[:, 1, 1] = lower.g, upper.g
            radial[:, 0, 1] = radial[:, 1, 0] = coupling.g
            drdx = np.zeros((len(q), 3, 3, 3))
            for k, (a, b) in enumerate(pairs):
                unit = vectors[:, k]/distances[:, k, None]
                drdx[:, k, a], drdx[:, k, b] = unit, -unit
            canonical_gradient = np.einsum("bijr,brac->bijac", radial, drdx)
            gradients = np.empty_like(canonical_gradient)
            for canonical, original in enumerate(self._order):
                gradients[..., original, :] = canonical_gradient[..., canonical, :]
        return validate_diabatic_result(DiabaticResult(matrix, gradients), q.shape, self.nstates)
