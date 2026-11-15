"""Complex and high-dimensional model potentials for subspace & instanton-anchor validation.

Covers four distinct chemical PES topologies beyond simple symmetric 2D valleys:

1. ``TwistedCurved`` + ``RotatedEmbedding`` (e.g. 10D ``twisted_10d``):
   3D non-planar reaction valley combining strong symmetric planar corner cutting
   in ``y`` with antisymmetric out-of-plane torsion and intermediate-frequency
   mode softening in ``z`` (``omega_z ~ 0.63 * omega_b``), embedded in 10D via
   a dense orthogonal rotation ``Q in SO(10)``.
2. ``PreReactionWell`` + ``RotatedEmbedding`` (e.g. 8D ``prereaction_well_8d``):
   Asymptotic free reactant channel -> localized pre-reaction van der Waals well ->
   curved central activation barrier with transverse mode modulation -> exothermic
   product channel, embedded and rotated in 8D.
3. ``PreReactionComplex`` + ``RotatedEmbedding`` (e.g. 8D ``prereaction_complex_8d``
   and ``submerged_complex_6d``):
   Deep pre-reaction complex + curved low or submerged central transfer barrier +
   deep post-reaction complex well (terminating inside the product complex),
   embedded and rotated in high dimensions.
4. ``LepsIonComplex9D`` (``leps_ion_complex_9d``):
   Full 9D 3-atom Cartesian PES in 3D space (LEPS atom transfer + damped
   ``-C_ion / (R^4 + R0^4)`` pre- and post-reaction ion--induced-dipole complexes)
   with exact analytic 9D forces, 5/6 rigid-body translations & rotations, and the
   physical ``RPMDrateCoordinate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from pyrpmd.analysis.subspace_paths import refine_saddle
from pyrpmd.constants import KB_AU
from pyrpmd.cv import RPMDrateCoordinate

CM = 1.0 / 219474.6313632
AMU = 1822.888486


@dataclass(frozen=True)
class TwistedCurved:
    """Non-planar 3D curved valley with offset U-bends in ``y`` and slow torsion/bend in ``z``."""

    V0: float = 0.016
    omega_barrier: float = 1500.0 * CM
    omega_y: float = 1650.0 * CM
    b_y: float = 13.0
    y_offset: float = -4.0
    c_y: float = 16.0
    omega_z: float = 1050.0 * CM
    soften_z: float = 0.22
    b_z: float = 11.0
    z_offset: float = 6.0
    c_z: float = 15.0

    @property
    def dimension(self) -> int:
        return 3

    @property
    def a(self) -> float:
        return float(np.sqrt(2.0 * self.V0) / self.omega_barrier)

    def _parts(self, q):
        q = np.asarray(q, dtype=float)
        x, y, z = q[..., 0], q[..., 1], q[..., 2]
        u = x / self.a
        sech2 = 1.0 / np.cosh(u) ** 2
        tanh_u = np.tanh(u)

        gy = self.b_y * np.exp(-((x - self.y_offset) / self.c_y) ** 2)
        dgy = -2.0 * (x - self.y_offset) / (self.c_y**2) * gy

        hz = self.b_z * np.exp(-((x - self.z_offset) / self.c_z) ** 2)
        dhz = -2.0 * (x - self.z_offset) / (self.c_z**2) * hz

        wz = self.omega_z * (1.0 - self.soften_z * sech2)
        dwz = self.omega_z * self.soften_z * (2.0 / self.a) * sech2 * tanh_u

        dy = y - gy
        dz = z - hz
        val = self.V0 * sech2 + 0.5 * self.omega_y**2 * dy**2 + 0.5 * wz**2 * dz**2
        gx = (
            -(2.0 * self.V0 / self.a) * sech2 * tanh_u
            - self.omega_y**2 * dy * dgy
            + wz * dwz * dz**2
            - wz**2 * dz * dhz
        )
        gy_grad = self.omega_y**2 * dy
        gz_grad = wz**2 * dz
        return val, np.stack([gx, gy_grad, gz_grad], axis=-1)

    def energy(self, q):
        return self._parts(q)[0]

    def gradient(self, q):
        return self._parts(q)[1]


@dataclass(frozen=True)
class PreReactionWell:
    """2D curved barrier preceded by a localized pre-reaction van der Waals well."""

    V0: float = 0.016
    omega_barrier: float = 1500.0 * CM
    well_depth: float = 0.0045
    well_center: float = -48.0
    well_width: float = 13.0
    product_drop: float = 0.0040
    product_center: float = 48.0
    product_width: float = 16.0
    omega_y: float = 1550.0 * CM
    soften_y: float = 0.20
    stiffen_well: float = 0.10
    b_y: float = 13.5
    c_y: float = 17.0

    @property
    def dimension(self) -> int:
        return 2

    @property
    def a(self) -> float:
        return float(np.sqrt(2.0 * self.V0) / self.omega_barrier)

    def _parts(self, q):
        q = np.asarray(q, dtype=float)
        x, y = q[..., 0], q[..., 1]
        u = x / self.a
        sech2 = 1.0 / np.cosh(u) ** 2
        dsech2 = -(2.0 / self.a) * sech2 * np.tanh(u)

        ew = np.exp(-((x - self.well_center) / self.well_width) ** 2)
        dew = -2.0 * (x - self.well_center) / (self.well_width**2) * ew

        tp = np.tanh((x - self.product_center) / self.product_width)
        sw = 0.5 * (1.0 + tp)
        dsw = 0.5 / (self.product_width * np.cosh((x - self.product_center) / self.product_width) ** 2)

        v_long = self.V0 * sech2 - self.well_depth * ew - self.product_drop * sw
        dv_long = self.V0 * dsech2 - self.well_depth * dew - self.product_drop * dsw

        gy = self.b_y * np.exp(-(x / self.c_y) ** 2)
        dgy = -2.0 * x / (self.c_y**2) * gy

        wy = self.omega_y * (1.0 - self.soften_y * sech2 + self.stiffen_well * ew)
        dwy = self.omega_y * (-self.soften_y * dsech2 + self.stiffen_well * dew)

        dy = y - gy
        val = v_long + 0.5 * wy**2 * dy**2
        gx = dv_long + wy * dwy * dy**2 - wy**2 * dy * dgy
        gy_grad = wy**2 * dy
        return val, np.stack([gx, gy_grad], axis=-1)

    def energy(self, q):
        return self._parts(q)[0]

    def gradient(self, q):
        return self._parts(q)[1]


@dataclass(frozen=True)
class PreReactionComplex:
    """2D curved reaction valley with deep pre- and post-reaction complexes.

    With ``submerged=False`` the central saddle sits moderately above the
    separated reactant asymptote (+0.006 Eh) while lying ~0.015 Eh above the
    pre-reaction complex; with ``submerged=True`` the saddle lies *below* the
    reactant asymptote (V_TS < 0).  On the product side a deep post-reaction
    complex (-0.024 Eh) commits products inside the complex well.
    """

    V0: float = 0.0155
    omega_barrier: float = 1500.0 * CM
    pre_depth: float = 0.0095
    pre_center: float = -56.0
    pre_width: float = 16.0
    post_depth: float = 0.0230
    post_center: float = 60.0
    post_width: float = 18.0
    exo_drop: float = 0.0140
    exo_center: float = 92.0
    exo_width: float = 16.0
    submerged: bool = False
    omega_y: float = 1450.0 * CM
    soften_y: float = 0.20
    stiffen_complex: float = 0.15
    b_y: float = 16.0
    c_y: float = 15.5

    @property
    def dimension(self) -> int:
        return 2

    @property
    def a(self) -> float:
        return float(np.sqrt(2.0 * self.V0) / self.omega_barrier)

    def _parts(self, q):
        q = np.asarray(q, dtype=float)
        x, y = q[..., 0], q[..., 1]
        u = x / self.a
        sech2 = 1.0 / np.cosh(u) ** 2
        dsech2 = -(2.0 / self.a) * sech2 * np.tanh(u)

        epre = np.exp(-((x - self.pre_center) / self.pre_width) ** 2)
        depre = -2.0 * (x - self.pre_center) / (self.pre_width**2) * epre

        epost = np.exp(-((x - self.post_center) / self.post_width) ** 2)
        depost = -2.0 * (x - self.post_center) / (self.post_width**2) * epost

        # Optional broad ion-dipole-like complex envelope that submerges the central TS below 0.
        sub_amp = 0.0185 if self.submerged else 0.0060
        esub = np.exp(-((x - 2.0) / 50.0) ** 2)
        desub = -2.0 * (x - 2.0) / (50.0**2) * esub

        sw_exo = 0.5 * (1.0 + np.tanh((x - self.exo_center) / self.exo_width))
        dsw_exo = 0.5 / (self.exo_width * np.cosh((x - self.exo_center) / self.exo_width) ** 2)

        v_long = (
            self.V0 * sech2
            - self.pre_depth * epre
            - self.post_depth * epost
            - sub_amp * esub
            - self.exo_drop * sw_exo
        )
        dv_long = (
            self.V0 * dsech2
            - self.pre_depth * depre
            - self.post_depth * depost
            - sub_amp * desub
            - self.exo_drop * dsw_exo
        )

        gy = self.b_y * np.exp(-((x + 3.0) / self.c_y) ** 2)
        dgy = -2.0 * (x + 3.0) / (self.c_y**2) * gy

        wy = self.omega_y * (1.0 - self.soften_y * sech2 + self.stiffen_complex * (epre + epost))
        dwy = self.omega_y * (-self.soften_y * dsech2 + self.stiffen_complex * (depre + depost))

        dy = y - gy
        val = v_long + 0.5 * wy**2 * dy**2
        gx = dv_long + wy * dwy * dy**2 - wy**2 * dy * dgy
        gy_grad = wy**2 * dy
        return val, np.stack([gx, gy_grad], axis=-1)

    def energy(self, q):
        return self._parts(q)[0]

    def gradient(self, q):
        return self._parts(q)[1]


@lru_cache(maxsize=64)
def _orthogonal_rotation(dimension: int, seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(dimension, dimension)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    q.setflags(write=False)
    return q


@dataclass(frozen=True)
class RotatedEmbedding:
    """Embed ``base`` with harmonic bath modes and rotate all coordinates by ``Q in SO(D)``."""

    base: object
    frequencies: tuple[float, ...]
    seed: int = 17

    @property
    def dimension(self) -> int:
        return int(self.base.dimension) + len(self.frequencies)

    @property
    def rotation(self) -> np.ndarray:
        return _orthogonal_rotation(self.dimension, int(self.seed))

    def _split(self, x):
        u = np.asarray(x, dtype=float) @ self.rotation
        d0 = int(self.base.dimension)
        return u[..., :d0], u[..., d0:]

    def embed(self, u_core):
        u_core = np.asarray(u_core, dtype=float)
        zeros = np.zeros(u_core.shape[:-1] + (len(self.frequencies),), dtype=float)
        return np.concatenate([u_core, zeros], axis=-1) @ self.rotation.T

    def energy(self, x):
        core, bath = self._split(x)
        omega = np.asarray(self.frequencies, dtype=float)
        return self.base.energy(core) + 0.5 * np.sum((omega**2) * (bath**2), axis=-1)

    def gradient(self, x):
        core, bath = self._split(x)
        omega = np.asarray(self.frequencies, dtype=float)
        g_core = self.base.gradient(core)
        g_bath = (omega**2) * bath
        return np.concatenate([g_core, g_bath], axis=-1) @ self.rotation.T


@dataclass(frozen=True)
class LepsIonComplex9D:
    """9D 3-atom Cartesian LEPS + damped ion--induced-dipole pre-reaction complex with analytic forces."""

    ion_strength: float = 2.8
    ion_damping: float = 180.0
    ion_pairs: tuple[str, ...] = ("01",)
    k_align: float = 0.045
    gamma_align: float = 0.35
    masses_amu: tuple[float, float, float] = (1.007825, 3.016049, 1.007825)
    D_e: float = 0.1744
    alpha: float = 1.04435
    r0: float = 1.40083
    sato: float = 0.106
    r_inf: float = 20.0

    @property
    def dimension(self) -> int:
        return 9

    @property
    def masses_au(self) -> np.ndarray:
        return np.asarray(self.masses_amu, dtype=float) * AMU

    @property
    def cartesian_masses(self) -> np.ndarray:
        return np.repeat(self.masses_au, 3)

    def _parts(self, x):
        arr = np.asarray(x, dtype=float)
        scalar = arr.ndim == 1 or (arr.ndim == 2 and arr.shape == (3, 3))
        g = arr.reshape(-1, 3, 3)
        v01 = g[:, 0] - g[:, 1]
        v12 = g[:, 1] - g[:, 2]
        v02 = g[:, 0] - g[:, 2]
        r01 = np.linalg.norm(v01, axis=-1)
        r12 = np.linalg.norm(v12, axis=-1)
        r02 = np.linalg.norm(v02, axis=-1)
        r = np.stack([r01, r12, r02], axis=-1)

        e = np.exp(-self.alpha * (r - self.r0))
        de_dr = -self.alpha * e
        c_scale = self.D_e / (4.0 * (1.0 + self.sato))
        Q = c_scale * ((3.0 + self.sato) * e**2 - (2.0 + 6.0 * self.sato) * e)
        dQ_dr = c_scale * (2.0 * (3.0 + self.sato) * e - (2.0 + 6.0 * self.sato)) * de_dr
        J = c_scale * ((1.0 + 3.0 * self.sato) * e**2 - (6.0 + 2.0 * self.sato) * e)
        dJ_dr = c_scale * (2.0 * (1.0 + 3.0 * self.sato) * e - (6.0 + 2.0 * self.sato)) * de_dr

        diff01 = J[:, 0] - J[:, 1]
        diff12 = J[:, 1] - J[:, 2]
        diff20 = J[:, 2] - J[:, 0]
        rad = np.maximum(0.5 * (diff01**2 + diff12**2 + diff20**2), 1e-30)
        root = np.sqrt(rad)
        val = np.sum(Q, axis=-1) - root

        droot_dJ0 = (2.0 * J[:, 0] - J[:, 1] - J[:, 2]) / (2.0 * root)
        droot_dJ1 = (2.0 * J[:, 1] - J[:, 0] - J[:, 2]) / (2.0 * root)
        droot_dJ2 = (2.0 * J[:, 2] - J[:, 0] - J[:, 1]) / (2.0 * root)
        dV_dr = dQ_dr - np.stack([droot_dJ0, droot_dJ1, droot_dJ2], axis=-1) * dJ_dr

        u01 = v01 / r01[:, None]
        u12 = v12 / r12[:, None]
        u02 = v02 / r02[:, None]
        grad = np.zeros_like(g)
        grad[:, 0] += dV_dr[:, 0, None] * u01 + dV_dr[:, 2, None] * u02
        grad[:, 1] += -dV_dr[:, 0, None] * u01 + dV_dr[:, 1, None] * u12
        grad[:, 2] += -dV_dr[:, 1, None] * u12 - dV_dr[:, 2, None] * u02

        if self.ion_strength:
            pair_map = {"01": (r01, u01, 0, 1), "12": (r12, u12, 1, 2)}
            for key in self.ion_pairs:
                r_bond, u_vec, i_a, i_b = pair_map[key]
                r2 = r_bond**2
                denom = r2**2 + self.ion_damping
                val = val - self.ion_strength / denom
                d_ion = (4.0 * self.ion_strength * (r_bond**3) / (denom**2))[:, None] * u_vec
                grad[:, i_a] += d_ion
                grad[:, i_b] -= d_ion

        if self.k_align:
            cos_theta = np.sum(u01 * u12, axis=-1)
            one_minus_c = 1.0 - cos_theta
            env = self.k_align * np.exp(-self.gamma_align * (r01 + r12 - 2.0 * self.r0))
            val = val + env * one_minus_c
            dv01 = (
                (-self.gamma_align * env * one_minus_c)[:, None] * u01
                - (env / r01)[:, None] * (u12 - cos_theta[:, None] * u01)
            )
            dv12 = (
                (-self.gamma_align * env * one_minus_c)[:, None] * u12
                - (env / r12)[:, None] * (u01 - cos_theta[:, None] * u12)
            )
            grad[:, 0] += dv01
            grad[:, 1] += -dv01 + dv12
            grad[:, 2] -= dv12

        if scalar:
            return float(val[0]), grad[0].ravel()
        return val, grad.reshape(arr.shape)

    def energy(self, x):
        return self._parts(x)[0]

    def gradient(self, x):
        return self._parts(x)[1]


def abstract_rate_run(
    potential,
    temperature_kelvin: float,
    nbeads: int,
    *,
    reference_distance: float = 165.0,
    guess_perturbation: float = 0.25,
    root=None,
):
    """Build a RateRun-compatible abstract-coordinate run (``rigid_body = False``)."""
    dim = int(potential.dimension)
    core = potential.base if isinstance(potential, RotatedEmbedding) else potential
    core_guess = np.zeros(int(core.dimension), dtype=float)
    if hasattr(core, "b_y"):
        y_off = float(getattr(core, "y_offset", 0.0))
        core_guess[1] = float(core.b_y) * np.exp(-((0.0 - y_off) / float(core.c_y)) ** 2)
    if core.dimension >= 3 and hasattr(core, "b_z"):
        z_off = float(getattr(core, "z_offset", 0.0))
        core_guess[2] = float(core.b_z) * np.exp(-((0.0 - z_off) / float(core.c_z)) ** 2)
    saddle_core, rec = refine_saddle(
        lambda u: core.gradient(u), core_guess, np.ones(int(core.dimension)),
        rigid_body=False, gradient_tolerance=1e-9,
    )
    if not rec["saddle"]:
        raise RuntimeError(f"failed to locate core saddle: {rec}")
    ts = potential.embed(saddle_core) if isinstance(potential, RotatedEmbedding) else saddle_core
    e0_core = np.zeros(int(core.dimension), dtype=float)
    e0_core[0] = 1.0
    direction = potential.embed(e0_core) if isinstance(potential, RotatedEmbedding) else e0_core

    def xi_value_gradient(x):
        vec = np.asarray(x, dtype=float).ravel()
        val = 1.0 + float((vec - ts) @ direction) / float(reference_distance)
        return val, direction / float(reference_distance)

    guess = ts + float(guess_perturbation) * np.cos(np.arange(dim, dtype=float) + 0.3)
    return SimpleNamespace(
        root=root,
        provider=None,
        beta=1.0 / (KB_AU * float(temperature_kelvin)),
        nbeads=int(nbeads),
        rigid_body=False,
        xi_star=1.0,
        saddle=ts,
        cartesian_masses=np.ones(dim, dtype=float),
        system=SimpleNamespace(geometry=guess, masses=np.ones(dim, dtype=float)),
        energy=lambda x: float(potential.energy(np.asarray(x, dtype=float))),
        gradient=lambda x: np.asarray(potential.gradient(np.asarray(x, dtype=float)), dtype=float).ravel(),
        xi_value_gradient=xi_value_gradient,
        xi=lambda x: xi_value_gradient(x)[0],
        xi_gradient=lambda x: xi_value_gradient(x)[1],
    )


def molecular_leps_rate_run(
    pes: LepsIonComplex9D,
    temperature_kelvin: float,
    nbeads: int,
    *,
    root=None,
):
    """Build a 9D 3-atom Cartesian RateRun with off-saddle bent TS guess and RPMDrateCoordinate."""
    bent_guess = np.array([0.0, 0.0, -1.8, 0.0, 0.0, 0.02, 0.05, 0.0, 1.75], dtype=float)
    saddle, rec = refine_saddle(pes.gradient, bent_guess, pes.cartesian_masses, rigid_body=True)
    if not rec["saddle"]:
        raise RuntimeError(f"failed to refine 9D LEPS saddle: {rec}")
    coord = RPMDrateCoordinate(
        [[0], [1, 2]], pes.masses_au, saddle.reshape(3, 3), [[0, 1]], [[1, 2]], pes.r_inf
    )

    def xi_value_gradient(x):
        val, deriv = coord.centroid_value_gradient(np.asarray(x, dtype=float).reshape(3, 3))[:2]
        return float(np.asarray(val)), np.asarray(deriv, dtype=float).ravel()

    near = saddle + np.array([0.0, 0.03, 0.04, 0.0, 0.0, -0.02, 0.0, -0.03, 0.05], dtype=float)
    return SimpleNamespace(
        root=root,
        provider=None,
        beta=1.0 / (KB_AU * float(temperature_kelvin)),
        nbeads=int(nbeads),
        rigid_body=True,
        xi_star=1.0,
        saddle=saddle,
        cartesian_masses=pes.cartesian_masses,
        system=SimpleNamespace(masses=pes.masses_au, geometry=near.reshape(3, 3)),
        energy=lambda x: float(pes.energy(np.asarray(x, dtype=float).ravel())),
        gradient=lambda x: np.asarray(pes.gradient(np.asarray(x, dtype=float).ravel()), dtype=float).ravel(),
        xi_value_gradient=xi_value_gradient,
        xi=lambda x: xi_value_gradient(x)[0],
        xi_gradient=lambda x: xi_value_gradient(x)[1],
    )


def complex_suite() -> dict[str, object]:
    """Named high-dimensional complex PES suite."""
    bath_7 = (550 * CM, 850 * CM, 1250 * CM, 1650 * CM, 2150 * CM, 2750 * CM, 3350 * CM)
    bath_6 = (600 * CM, 1050 * CM, 1550 * CM, 2100 * CM, 2650 * CM, 3200 * CM)
    return {
        "twisted_3d": TwistedCurved(),
        "twisted_10d": RotatedEmbedding(TwistedCurved(), bath_7, seed=19),
        "prereaction_well_2d": PreReactionWell(),
        "prereaction_well_8d": RotatedEmbedding(PreReactionWell(), bath_6, seed=23),
        "prereaction_complex_2d": PreReactionComplex(submerged=False),
        "prereaction_complex_8d": RotatedEmbedding(PreReactionComplex(submerged=False), bath_6, seed=29),
        "submerged_complex_6d": RotatedEmbedding(
            PreReactionComplex(submerged=True), (700 * CM, 1300 * CM, 2000 * CM, 2800 * CM), seed=31
        ),
        "leps_ion_complex_9d": LepsIonComplex9D(ion_strength=2.8, ion_damping=180.0),
    }
