"""Analytic diabatic model Hamiltonians with positive Cartesian gradients.

Coordinates are bohr, energies hartree, and hbar=1.  Models provide the full
matrix, including the common harmonic energy on every diagonal.  An active
coordinate index is ``3 * atom + axis`` in an atom-major flattened geometry;
these models do not convert mass-weighted normal coordinates automatically.

Tully defaults follow J. C. Tully, JCP 93, 1061 (1990), Eqs. (21), (23), (24),
doi:10.1063/1.459170.  These model potentials do not choose an initial ensemble,
propagator, electronic estimator, or a converged scattering calculation.
"""

from collections.abc import Mapping

import numpy as np

from .base import Potential, PotentialError, coordinates as validate_coordinates


__all__ = [
    "TullyModelOne", "TullyModelTwo", "TullyModelThree",
    "LinearVibronicCoupling", "DiscreteSpinBoson", "model_from_dict",
]

_SCHEMA = "pyrpmd.nonadiabatic_model"
_VERSION = 1


def _integer(value, name, *, minimum=0):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(value, name, *, nonnegative=False):
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind not in "fiu" or not np.isfinite(raw):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _immutable(value, dtype=None):
    """Own the parameters; an immutable bytes buffer also prevents setflags."""
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _vector(value, name, size, *, positive=False, nonnegative=False):
    raw = np.asarray(value)
    if raw.shape != (size,) or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must be a finite real vector of length {size}")
    if positive and np.any(raw <= 0):
        raise ValueError(f"{name} must be positive")
    if nonnegative and np.any(raw < 0):
        raise ValueError(f"{name} must be nonnegative")
    return _immutable(raw, float)


def _hermitian(value, name, ndim):
    raw = np.asarray(value)
    if (raw.ndim != ndim or raw.dtype.kind not in "fciu" or min(raw.shape) < 1
            or raw.shape[-1] != raw.shape[-2] or not np.isfinite(raw).all()):
        shape = "[S,S]" if ndim == 2 else "[D,S,S]"
        raise ValueError(f"{name} must be a finite numerical {shape} array")
    adjoint = raw.swapaxes(-1, -2).conj()
    if not np.allclose(raw, adjoint, rtol=0, atol=1e-12):
        raise ValueError(f"{name} must be Hermitian in its electronic indices")
    # Only roundoff-level departures are symmetrized, with an absolute tolerance
    # so that a large common diagonal cannot conceal a non-Hermitian coupling.
    return _immutable(0.5 * raw + 0.5 * adjoint,
                      complex if np.iscomplexobj(raw) else float)


def _coordinate_layout(indices, count, natoms, *, separate_atoms=False):
    if indices is None:
        raw = np.arange(count, dtype=int) * (3 if separate_atoms else 1)
    else:
        raw = np.asarray(indices)
    if (raw.shape != (count,) or raw.dtype.kind not in "iu" or np.any(raw < 0)
            or len(np.unique(raw)) != count):
        raise ValueError("coordinates must be distinct nonnegative integer indices, one per mode")
    required_atoms = int(raw.max()) // 3 + 1
    natoms = required_atoms if natoms is None else _integer(natoms, "natoms", minimum=1)
    if required_atoms > natoms:
        raise ValueError("active coordinate index is outside natoms * 3")
    return tuple(int(i) for i in raw), natoms


def _complex_payload(value):
    array = np.asarray(value)
    return {"real": array.real.tolist(), "imag": array.imag.tolist()}


def _complex_from_payload(value, name):
    if not isinstance(value, Mapping) or set(value) != {"real", "imag"}:
        raise ValueError(f"{name} must contain exactly real and imag")
    real, imag = np.asarray(value["real"]), np.asarray(value["imag"])
    if (real.shape != imag.shape or real.dtype.kind not in "fiu"
            or imag.dtype.kind not in "fiu" or not np.isfinite(real).all()
            or not np.isfinite(imag).all()):
        raise ValueError(f"{name} real/imag must be same-shape finite real arrays")
    return real.astype(float) + 1j * imag if np.any(imag != 0) else real.astype(float)


def _result(matrix, gradients):
    if not np.isfinite(matrix).all() or not np.isfinite(gradients).all():
        raise PotentialError("nonadiabatic model evaluation overflowed")
    # Delay this import: nonadiabatic itself imports the potentials package.
    from ..nonadiabatic import DiabaticResult
    return DiabaticResult(np.ascontiguousarray(matrix), np.ascontiguousarray(gradients))


class _MatrixModel(Potential):
    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("nonadiabatic model parameters are immutable")
        object.__setattr__(self, name, value)

    def _coordinates(self, q):
        q = validate_coordinates(q)
        if q.shape[1] != self.natoms:
            raise PotentialError(f"model requires coordinates [B,{self.natoms},3]")
        return q

    @property
    def metadata(self):
        """Detached, JSON-safe provider information and full model identity."""
        return {"provider": "nonadiabatic_model", "thread_safety": "reentrant",
                "execution_device": "cpu", "periodic": False,
                "energy_unit": "hartree", "length_unit": "bohr",
                "derivative": "gradient", "nstates": self.nstates,
                "natoms": self.natoms, "model_identity": self.to_dict()}

    def to_dict(self):
        """Versioned identity accepted by :func:`model_from_dict`."""
        return {"schema": _SCHEMA, "version": _VERSION,
                "model": self._model_name, "parameters": self._parameters_dict()}


class _TullyModel(_MatrixModel):
    nstates = 2
    ncoordinates = 1

    def _initialize(self, coordinate, natoms, parameters):
        coordinate = _integer(coordinate, "coordinate")
        self.coordinates, self.natoms = _coordinate_layout([coordinate], 1, natoms)
        self.coordinate = coordinate
        for key, value in parameters.items():
            setattr(self, key, value)
        self._parameter_names = tuple(parameters)
        self._frozen = True

    def _parameters_dict(self):
        return {"natoms": self.natoms, "coordinate": self.coordinate,
                **{key: getattr(self, key) for key in self._parameter_names}}

    def evaluate(self, q):
        q = self._coordinates(q)
        x = q.reshape(len(q), -1)[:, self.coordinate]
        with np.errstate(over="ignore", invalid="ignore"):
            v0, v1, coupling, g0, g1, gc = self._fields(x)
        matrix = np.empty((len(q), 2, 2))
        matrix[:, 0, 0], matrix[:, 1, 1] = v0, v1
        matrix[:, 0, 1] = matrix[:, 1, 0] = coupling
        gradients = np.zeros((len(q), 2, 2, self.natoms * 3))
        gradients[:, 0, 0, self.coordinate] = g0
        gradients[:, 1, 1, self.coordinate] = g1
        gradients[:, 0, 1, self.coordinate] = gradients[:, 1, 0, self.coordinate] = gc
        return _result(matrix, gradients.reshape(len(q), 2, 2, self.natoms, 3))


class TullyModelOne(_TullyModel):
    """Simple avoided crossing, Tully (1990) Eq. (21).

    ``V00 = a * sign(x) * (1 - exp(-b*abs(x)))``, ``V11 = -V00``,
    ``V01 = c*exp(-d*x*x)``.  Defaults are the paper's atomic-unit parameters.
    ``coordinate=0`` selects atom 0's x coordinate; all other gradients vanish.
    ``natoms=None`` infers the atom count from the selected coordinate.
    """
    _model_name = "tully1"

    def __init__(self, *, a=0.01, b=1.6, c=0.005, d=1.0, natoms=1, coordinate=0):
        self._initialize(coordinate, natoms, {
            "a": _real(a, "a"), "b": _real(b, "b", nonnegative=True),
            "c": _real(c, "c"), "d": _real(d, "d", nonnegative=True)})

    def _fields(self, x):
        decay = np.exp(-self.b * np.abs(x))
        v = self.a * np.sign(x) * (-np.expm1(-self.b * np.abs(x)))
        g = (self.a * decay) * self.b
        coupling = self.c * np.exp(-self.d * x * x)
        gc = (-2 * self.d * coupling) * x
        return v, -v, coupling, g, -g, gc


class TullyModelTwo(_TullyModel):
    """Dual avoided crossing, Tully (1990) Eq. (23).

    ``V00=0``, ``V11=-a*exp(-b*x*x)+e0``, ``V01=c*exp(-d*x*x)``.
    Coordinate and unit conventions match :class:`TullyModelOne`.
    """
    _model_name = "tully2"

    def __init__(self, *, a=0.10, b=0.28, c=0.015, d=0.06, e0=0.05,
                 natoms=1, coordinate=0):
        self._initialize(coordinate, natoms, {
            "a": _real(a, "a"), "b": _real(b, "b", nonnegative=True),
            "c": _real(c, "c"), "d": _real(d, "d", nonnegative=True),
            "e0": _real(e0, "e0")})

    def _fields(self, x):
        gaussian = self.a * np.exp(-self.b * x * x)
        v = self.e0 - gaussian
        g = (2 * self.b * gaussian) * x
        coupling = self.c * np.exp(-self.d * x * x)
        gc = (-2 * self.d * coupling) * x
        return np.zeros_like(x), v, coupling, np.zeros_like(x), g, gc


class TullyModelThree(_TullyModel):
    """Extended coupling with reflection, Tully (1990) Eq. (24).

    ``V00=a``, ``V11=-a``; ``V01=b*exp(c*x)`` for x<0 and
    ``b*(2-exp(-c*x))`` for x>=0.  The positive-x diabatic coupling tends to
    ``2*b``, not zero: asymptotic adiabatic and diabatic states differ.
    """
    _model_name = "tully3"

    def __init__(self, *, a=0.0006, b=0.10, c=0.90, natoms=1, coordinate=0):
        self._initialize(coordinate, natoms, {
            "a": _real(a, "a"), "b": _real(b, "b"),
            "c": _real(c, "c", nonnegative=True)})

    def _fields(self, x):
        decay = np.exp(-self.c * np.abs(x))
        coupling = self.b * np.where(x < 0, decay, 2 - decay)
        gc = (self.b * decay) * self.c
        zero = np.zeros_like(x)
        return np.full_like(x, self.a), np.full_like(x, -self.a), coupling, zero, zero, gc


class LinearVibronicCoupling(_MatrixModel):
    """General fixed-basis Hermitian linear vibronic-coupling model.

    For active coordinates ``Q[d]=q.flat[coordinates[d]]-reference[d]``,
    ``V=h0 + sum_d linear[d]*Q[d] + I/2 * sum_d harmonic[d]*Q[d]**2``.
    ``h0[S,S]`` and ``linear[D,S,S]`` are Hermitian; D and S may be any
    positive integers.  ``harmonic[D]`` contains nonnegative force constants
    (hartree/bohr**2), not frequencies.  There is no mass-weighted-coordinate
    conversion.  Default active indices are the first D flattened Cartesian
    coordinates; natoms is inferred if omitted.  Inactive coordinates are free.
    """
    _model_name = "lvc"

    def __init__(self, h0, linear, *, harmonic=None, reference=None,
                 coordinates=None, natoms=None):
        self.h0 = _hermitian(h0, "h0", 2)
        self.linear = _hermitian(linear, "linear", 3)
        self.nstates = len(self.h0)
        if self.linear.shape[1:] != self.h0.shape:
            raise ValueError("linear electronic dimensions must match h0")
        self.ncoordinates = len(self.linear)
        self.harmonic = _vector(np.zeros(self.ncoordinates) if harmonic is None else harmonic,
                                "harmonic", self.ncoordinates, nonnegative=True)
        self.reference = _vector(np.zeros(self.ncoordinates) if reference is None else reference,
                                "reference", self.ncoordinates)
        self.coordinates, self.natoms = _coordinate_layout(coordinates, self.ncoordinates, natoms)
        self._frozen = True

    def evaluate(self, q):
        q = self._coordinates(q)
        active = q.reshape(len(q), -1)[:, self.coordinates] - self.reference
        identity = np.eye(self.nstates)
        with np.errstate(over="ignore", invalid="ignore"):
            common = 0.5 * np.einsum("bd,d,bd->b", active, self.harmonic, active)
            matrix = (self.h0 + np.einsum("bd,dij->bij", active, self.linear)
                      + common[:, None, None] * identity)
            active_gradient = (self.linear[None]
                               + (active * self.harmonic)[..., None, None] * identity)
        gradients = np.zeros((len(q), self.nstates, self.nstates, self.natoms * 3),
                             dtype=matrix.dtype)
        gradients[..., self.coordinates] = active_gradient.transpose(0, 2, 3, 1)
        return _result(matrix, gradients.reshape(len(q), self.nstates, self.nstates, self.natoms, 3))

    def _parameters_dict(self):
        return {"h0": _complex_payload(self.h0), "linear": _complex_payload(self.linear),
                "harmonic": self.harmonic.tolist(), "reference": self.reference.tolist(),
                "coordinates": list(self.coordinates), "natoms": self.natoms}


class DiscreteSpinBoson(_MatrixModel):
    """Two electronic states coupled to an explicit finite harmonic bath.

    ``V00=U+bias+sum(c*Q)``, ``V11=U-bias-sum(c*Q)``,
    ``V01=tunnelling``, ``V10=conj(tunnelling)``, where
    ``U=sum(masses*frequencies**2*Q**2)/2``.  Bias is half the diabatic gap
    at Q=0; tunnelling is the off-diagonal matrix element.  Frequencies are
    angular frequencies in inverse atomic time; masses are in electron masses;
    couplings have units hartree/bohr.  No counterterm is added and no spectral
    density is discretized.  All input bath arrays have length D>=1.

    By default each mode occupies one artificial atom's x coordinate, so
    ``coordinates=(0,3,...,3*(D-1))`` and ``natoms=D``.  Explicit Cartesian
    indices are also supported.  Modes assigned to one atom must have equal
    masses.  Use ``nuclear_masses[N]`` in the dynamics; unused atoms have mass
    one and all inactive coordinates are free spectator coordinates.
    """
    _model_name = "spin_boson"

    def __init__(self, frequencies, couplings, *, masses=None, bias=0.0, tunnelling=1.0,
                 coordinates=None, natoms=None, reference=None):
        raw = np.asarray(frequencies)
        if raw.ndim != 1 or raw.size < 1:
            raise ValueError("frequencies must be a nonempty vector")
        self.ncoordinates = len(raw)
        self.frequencies = _vector(raw, "frequencies", self.ncoordinates, positive=True)
        self.couplings = _vector(couplings, "couplings", self.ncoordinates)
        self.masses = _vector(np.ones(self.ncoordinates) if masses is None else masses,
                             "masses", self.ncoordinates, positive=True)
        self.bias = _real(bias, "bias")
        raw_tunnelling = np.asarray(tunnelling)
        if (raw_tunnelling.shape != () or raw_tunnelling.dtype.kind not in "fciu"
                or not np.isfinite(raw_tunnelling)):
            raise ValueError("tunnelling must be a finite numerical scalar")
        self.tunnelling = (complex(raw_tunnelling) if np.iscomplexobj(raw_tunnelling)
                           else float(raw_tunnelling))
        self.coordinates, self.natoms = _coordinate_layout(
            coordinates, self.ncoordinates, natoms, separate_atoms=True)
        nuclear_masses = np.ones(self.natoms)
        assigned = {}
        for index, mass in zip(self.coordinates, self.masses):
            atom = index // 3
            if atom in assigned and assigned[atom] != mass:
                raise ValueError("modes on the same atom must have the same mass")
            assigned[atom] = mass
            nuclear_masses[atom] = mass
        self.nuclear_masses = _immutable(nuclear_masses)
        h0 = np.array([[self.bias, self.tunnelling],
                       [np.conj(self.tunnelling), -self.bias]])
        linear = np.zeros((self.ncoordinates, 2, 2))
        linear[:, 0, 0], linear[:, 1, 1] = self.couplings, -self.couplings
        with np.errstate(over="ignore", invalid="ignore"):
            harmonic = self.masses * self.frequencies**2
        self._lvc = LinearVibronicCoupling(h0, linear, harmonic=harmonic, reference=reference,
                                         coordinates=self.coordinates, natoms=self.natoms)
        self.harmonic, self.reference = self._lvc.harmonic, self._lvc.reference
        self.nstates = 2
        self._frozen = True

    def evaluate(self, q):
        return self._lvc.evaluate(q)

    def _parameters_dict(self):
        return {"frequencies": self.frequencies.tolist(), "couplings": self.couplings.tolist(),
                "masses": self.masses.tolist(), "bias": self.bias,
                "tunnelling": _complex_payload(self.tunnelling),
                "reference": self.reference.tolist(), "coordinates": list(self.coordinates),
                "natoms": self.natoms}


def model_from_dict(value):
    """Rebuild one of these models from its versioned, JSON-safe identity.

    Only the five builtin model names are accepted.  No import string, pickle,
    executable code, optional keys, or unrecognized version is interpreted.
    Provider ``metadata`` contains this identity under ``model_identity``.
    """
    if not isinstance(value, Mapping) or set(value) != {"schema", "version", "model", "parameters"}:
        raise ValueError("model identity must contain exactly schema, version, model, parameters")
    if (value["schema"] != _SCHEMA or type(value["version"]) is not int
            or value["version"] != _VERSION):
        raise ValueError("unsupported nonadiabatic model schema/version")
    definitions = {
        "tully1": (TullyModelOne, {"a", "b", "c", "d", "coordinate", "natoms"}),
        "tully2": (TullyModelTwo, {"a", "b", "c", "d", "e0", "coordinate", "natoms"}),
        "tully3": (TullyModelThree, {"a", "b", "c", "coordinate", "natoms"}),
        "lvc": (LinearVibronicCoupling, {"h0", "linear", "harmonic", "reference", "coordinates", "natoms"}),
        "spin_boson": (DiscreteSpinBoson, {"frequencies", "couplings", "masses", "bias",
                                          "tunnelling", "reference", "coordinates", "natoms"}),
    }
    name = value["model"]
    if not isinstance(name, str) or name not in definitions:
        raise ValueError("unknown nonadiabatic model")
    factory, expected = definitions[name]
    if not isinstance(value["parameters"], Mapping) or set(value["parameters"]) != expected:
        raise ValueError(f"{name} model parameters must contain exactly {sorted(expected)}")
    parameters = dict(value["parameters"])
    for key in ("h0", "linear") if name == "lvc" else ("tunnelling",) if name == "spin_boson" else ():
        parameters[key] = _complex_from_payload(parameters[key], key)
    return factory(**parameters)
