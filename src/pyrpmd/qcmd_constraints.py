"""Conditional imaginary-time sampling at fixed triatomic quasi-centroids.

This is a configurational Metropolis sampler, not real-time QCMD dynamics.
It samples one isolated, nonperiodic bent triatomic with fixed bead means of
both bond lengths, the bond angle and the molecular center of mass. Bead
orientations and center-of-mass internal modes remain in the sampled ring.

The Cartesian ring density is transformed directly to bond lengths, angle,
Haar orientations and molecular COM coordinates. Its volume Jacobian is
``prod_b r1[b]**2 * r2[b]**2 * sin(theta[b])``. The mean constraints are
linear in these coordinates; their constant delta-function normalization
cancels in Metropolis ratios. Consequently no uncorrected Cartesian
constraint multiplier is used as a quasi-centroid mean force.
"""

from __future__ import annotations

import copy
import json
import re

import numpy as np
from scipy.spatial.transform import Rotation

from .checkpoint import atomic_savez, load_checkpoint
from .cmd import _integer, _real, _scalar
from .io import fingerprint_payload, jsonable
from .potentials.base import coordinates, validate_result
from .qcmd_geometry import TriatomicCoordinates


KIND = "PyRPMD.quasicentroid-conditional.v1"


class ConstrainedQuasiCentroidSampler:
    """Metropolis sampling of an isolated triatomic's conditional PIMD ring.

    ``positions`` has shape ``[bead,3,3]``. ``molecule`` gives atom indices
    ``(central,terminal1,terminal2)``. ``target`` is ``(r1,r2,theta)`` in bohr
    and radians; if omitted it is the initial ring's mean. The initial ring
    must satisfy that target. The supplied PES must describe an isolated,
    translation-invariant molecule and must be deterministic.

    A step proposes one paired internal-coordinate displacement, one paired
    bead-COM displacement, or one bead rotation. Positive proposal widths
    are needed for sampling all unconstrained coordinates. An explicitly
    zero width freezes those coordinates and is recorded as an additional
    sampling restriction; no claim of ergodicity follows from this API.

    ``potential_identity`` is a caller-provided SHA256 fingerprint covering
    the potential parameters and source/model content. Checkpoints require
    this identity together with all sampling settings and include the exact
    state, counters and RNG state. No MC step is assigned a physical time.
    """

    def __init__(self, potential, masses, beta, positions, *, target=None,
                 molecule=(0, 1, 2), bond_step=.025, angle_step=.02,
                 rotation_step=.05, center_step=.025, seed=0,
                 potential_identity, cell=None, pbc=False):
        self.potential = potential
        self._geometry = TriatomicCoordinates([molecule], 3, cell=cell, pbc=pbc)
        self._indices = self._geometry.molecules[0]
        mass = _real(masses, "masses")
        if mass.shape != (3,) or np.any(mass <= 0):
            raise ValueError("conditional triatomic sampling requires three positive masses")
        self._masses = mass
        self._ordered_masses = mass[self._indices]
        self._beta = _scalar(beta, "beta")
        q = coordinates(positions).copy()
        if q.shape[1:] != (3, 3):
            raise ValueError("conditional sampling requires exactly one isolated triatomic")
        self._beads = len(q)
        if (not isinstance(potential_identity, str)
                or re.fullmatch(r"[0-9a-f]{64}", potential_identity) is None):
            raise ValueError("potential_identity must be a SHA256 fingerprint of potential parameters and source/model content")
        self._potential_identity = potential_identity
        bond = _real(bond_step, "bond_step")
        if bond.ndim == 0:
            bond = np.repeat(bond, 2)
        if bond.shape != (2,) or np.any(bond < 0):
            raise ValueError("bond_step must be one or two nonnegative proposal widths")
        self._internal_widths = np.r_[bond, _scalar(angle_step, "angle_step", positive=False)]
        self._rotation_width = _scalar(rotation_step, "rotation_step", positive=False)
        self._center_width = _scalar(center_step, "center_step", positive=False)
        self._move_types = []
        if self._beads > 1 and np.any(self._internal_widths > 0):
            self._move_types.append(0)
        if self._beads > 1 and self._center_width > 0:
            self._move_types.append(1)
        if self._rotation_width > 0:
            self._move_types.append(2)
        if not self._move_types:
            raise ValueError("at least one unconstrained coordinate must have a positive proposal width")
        self.rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        self._internal = self._geometry.values(q)[:, 0]
        self._target = (self._internal.mean(axis=0) if target is None
                        else _real(target, "target"))
        if (self._target.shape != (3,) or not self._valid_internal(self._target[None])
                or not np.allclose(self._internal.mean(axis=0), self._target, rtol=0., atol=2e-12)):
            raise ValueError("initial positions must satisfy the positive bond and bent-angle quasi-centroid target")
        ordered = q[:, self._indices]
        self._centers = np.einsum("a,bac->bc", self._ordered_masses, ordered) / self._masses.sum()
        self._fixed_center = self._centers.mean(axis=0)
        u = ordered[:, 1] - ordered[:, 0]
        v = ordered[:, 2] - ordered[:, 0]
        axis1 = u / np.linalg.norm(u, axis=1)[:, None]
        axis2 = v - np.sum(axis1 * v, axis=1)[:, None] * axis1
        axis2 /= np.linalg.norm(axis2, axis=1)[:, None]
        # Rows of the body-to-laboratory map are the molecular frame axes.
        frame = np.stack((axis1, axis2, np.cross(axis1, axis2)), axis=1)
        self._orientations = Rotation.from_matrix(frame).as_quat()
        self._q = self._cartesian(self._internal, self._orientations, self._centers)
        self._energies = self._evaluate(self._q)
        self._spring = self._spring_energy(self._q)
        self._log_weight = self._weight(self._internal, self._energies, self._spring)
        self._step = 0
        self._attempted = np.zeros(3, dtype=np.int64)
        self._accepted = np.zeros(3, dtype=np.int64)
        self._settings = self._scientific_settings()
        self._identity = fingerprint_payload(self._settings)

    @staticmethod
    def _valid_internal(value):
        return (np.isfinite(value).all() and np.all(value[:, :2] > 1e-12)
                and np.all(value[:, 2] > 1e-10) and np.all(value[:, 2] < np.pi - 1e-10))

    def _cartesian(self, internal, orientations, centers):
        body = np.zeros((len(internal), 3, 3))
        body[:, 1, 0] = internal[:, 0]
        body[:, 2, 0] = internal[:, 1] * np.cos(internal[:, 2])
        body[:, 2, 1] = internal[:, 1] * np.sin(internal[:, 2])
        body -= (np.einsum("a,bac->bc", self._ordered_masses, body) / self._masses.sum())[:, None]
        rotated = body @ Rotation.from_quat(orientations).as_matrix() + centers[:, None]
        q = np.empty_like(rotated)
        q[:, self._indices] = rotated
        return q

    def _evaluate(self, q):
        return validate_result(self.potential.evaluate(q), q.shape).energy.copy()

    def _spring_energy(self, q):
        difference = q - np.roll(q, -1, axis=0)
        return float(.5 * (self._beads / self._beta)**2
                     * np.sum(self._masses[None, :, None] * difference**2))

    def _weight(self, internal, energies, spring):
        # Configuration probability at beta_P=beta/P in the sum-bead convention.
        value = (-self._beta / self._beads * (float(energies.sum()) + spring)
                 + float(np.sum(2 * np.log(internal[:, 0]) + 2 * np.log(internal[:, 1])
                                + np.log(np.sin(internal[:, 2])))))
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite conditional PIMD log weight")
        return value

    def _scientific_settings(self):
        return {"format": KIND, "algorithm": "internal-coordinate-metropolis",
                "beta_au": self._beta, "beads": self._beads,
                "masses_au": self._masses.tolist(), "molecule": self._indices.tolist(),
                "target": self._target.tolist(), "fixed_centroid_com_bohr": self._fixed_center.tolist(),
                "internal_proposal_widths": self._internal_widths.tolist(),
                "rotation_proposal_width_radian": self._rotation_width,
                "center_proposal_width_bohr": self._center_width,
                "potential_identity": self._potential_identity,
                "measure": "Cartesian_PIMD_delta_mean_internal_delta_centroid_COM",
                "volume_jacobian": "product_beads(r1^2*r2^2*sin(theta))",
                "orientation_measure": "Haar_SO3", "periodic": False,
                "physical_dynamics": False}

    @property
    def positions(self):
        """A copy of the current constrained ring, in bohr."""
        return self._q.copy()

    @property
    def internal_coordinates(self):
        return self._internal.copy()

    @property
    def target(self):
        return self._target.copy()

    @property
    def step_count(self):
        return self._step

    @property
    def log_weight(self):
        """Unnormalized log density with respect to the declared coordinates."""
        return self._log_weight

    def metadata(self):
        """Scientific identity and conservative, purely numerical diagnostics."""
        metadata = copy.deepcopy(self._settings)
        metadata.update(identity=self._identity, steps=self._step,
                        attempted=self._attempted.tolist(), accepted=self._accepted.tolist(),
                        physical_converged=False, converged=False,
                        all_unconstrained_coordinates_mobile=bool(
                            self._rotation_width > 0 and (self._beads == 1 or
                            (self._center_width > 0 and np.all(self._internal_widths > 0)))))
        return metadata

    def step(self):
        """Make one reversible proposal; return whether it was accepted.

        A PES failure leaves the ring, counters and RNG exactly unchanged.
        An ordinary Metropolis rejection consumes the proposal RNG and counts
        as a completed MC step, but does not change the ring.
        """
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        try:
            move = self._move_types[int(self.rng.integers(len(self._move_types)))]
            if move in (0, 1):
                first = int(self.rng.integers(self._beads))
                second = int(self.rng.integers(self._beads - 1))
                second += second >= first
                selected = np.array([first, second])
            else:
                selected = np.array([int(self.rng.integers(self._beads))])
            internal = self._internal[selected].copy()
            orientation = self._orientations[selected].copy()
            centers = self._centers[selected].copy()
            if move == 0:
                delta = self.rng.normal(size=3) * self._internal_widths
                internal[0] += delta
                internal[1] -= delta
            elif move == 1:
                delta = self.rng.normal(size=3) * self._center_width
                centers[0] += delta
                centers[1] -= delta
            else:
                rotation = Rotation.from_rotvec(self.rng.normal(size=3) * self._rotation_width)
                orientation = (Rotation.from_quat(orientation) * rotation).as_quat()
            accept = False
            if self._valid_internal(internal):
                trial_q = self._q.copy()
                trial_q[selected] = self._cartesian(internal, orientation, centers)
                energies = self._energies.copy()
                energies[selected] = self._evaluate(trial_q[selected])
                spring = self._spring_energy(trial_q)
                full_internal = self._internal.copy()
                full_internal[selected] = internal
                weight = self._weight(full_internal, energies, spring)
                accept = np.log(self.rng.random()) < min(0., weight - self._log_weight)
        except BaseException:
            self.rng.bit_generator.state = rng_state
            raise
        if accept:
            self._internal[selected] = internal
            self._orientations[selected] = orientation
            self._centers[selected] = centers
            self._q = trial_q
            self._energies = energies
            self._spring = spring
            self._log_weight = weight
            self._accepted[move] += 1
        self._attempted[move] += 1
        self._step += 1
        return bool(accept)

    def sample(self, steps, *, interval=1):
        """Run MC proposals and return rings at multiples of ``interval``.

        The initial state is excluded. ``steps`` must be divisible by the
        interval, so no unsaved tail is silently discarded. Successive rings
        are generally correlated; equilibrium and effective sample size need
        independent checks.
        """
        steps = _integer(steps, "steps")
        interval = _integer(interval, "interval")
        if steps % interval:
            raise ValueError("steps must be divisible by interval")
        samples = np.empty((steps // interval, self._beads, 3, 3))
        for index in range(steps):
            self.step()
            if (index + 1) % interval == 0:
                samples[index // interval] = self._q
        return samples

    def save_checkpoint(self, path):
        """Atomically save the full conditional sampling state and RNG."""
        atomic_savez(path, kind=KIND, identity=self._identity,
                     settings=json.dumps(self._settings, sort_keys=True),
                     q=self._q, internal=self._internal, orientations=self._orientations,
                     centers=self._centers, energies=self._energies, spring=self._spring,
                     log_weight=self._log_weight, step=self._step,
                     attempted=self._attempted, accepted=self._accepted,
                     rng=json.dumps(jsonable(self.rng.bit_generator.state)))

    def load_checkpoint(self, path):
        """Validate before replacing state; a failed load is transactional."""
        with load_checkpoint(path, expected_kind=KIND, require_integrity=True) as saved:
            if (str(saved["identity"]) != self._identity
                    or fingerprint_payload(json.loads(str(saved["settings"]))) != self._identity):
                raise ValueError("conditional PIMD checkpoint scientific identity changed")
            arrays = {name: _real(saved[name], f"checkpoint {name}") for name in
                      ("q", "internal", "orientations", "centers", "energies")}
            shapes = {"q": (self._beads, 3, 3), "internal": (self._beads, 3),
                      "orientations": (self._beads, 4), "centers": (self._beads, 3),
                      "energies": (self._beads,)}
            if any(arrays[name].shape != shapes[name] for name in shapes):
                raise ValueError("conditional PIMD checkpoint state shape changed")
            internal, centers = arrays["internal"], arrays["centers"]
            if (not self._valid_internal(internal)
                    or not np.allclose(internal.mean(axis=0), self._target, rtol=0., atol=2e-11)
                    or not np.allclose(centers.mean(axis=0), self._fixed_center, rtol=0., atol=2e-11)
                    or not np.allclose(np.sum(arrays["orientations"]**2, axis=1), 1., rtol=0., atol=2e-12)):
                raise ValueError("conditional PIMD checkpoint constraints or orientations are invalid")
            q = self._cartesian(internal, arrays["orientations"], centers)
            if not np.allclose(q, arrays["q"], rtol=0., atol=2e-12):
                raise ValueError("conditional PIMD checkpoint Cartesian/internal geometries disagree")
            actual_energies = self._evaluate(arrays["q"])
            if not np.allclose(actual_energies, arrays["energies"], rtol=2e-12, atol=2e-12):
                raise ValueError("conditional PIMD checkpoint potential energies changed")
            spring = _real(saved["spring"], "checkpoint spring")
            weight = _real(saved["log_weight"], "checkpoint log_weight")
            if (spring.ndim or weight.ndim or
                    not np.isclose(float(spring), self._spring_energy(arrays["q"]), rtol=2e-12, atol=2e-12)
                    or not np.isclose(float(weight), self._weight(internal, arrays["energies"], float(spring)),
                                      rtol=2e-12, atol=2e-12)):
                raise ValueError("conditional PIMD checkpoint action is inconsistent")
            step_value = np.asarray(saved["step"])
            if step_value.ndim:
                raise ValueError("conditional PIMD checkpoint step must be a scalar integer")
            step = _integer(step_value.item(), "checkpoint step", 0)
            attempted, accepted = np.asarray(saved["attempted"]), np.asarray(saved["accepted"])
            if (attempted.shape != (3,) or accepted.shape != (3,)
                    or attempted.dtype.kind not in "iu" or accepted.dtype.kind not in "iu"
                    or np.any(attempted < 0) or np.any(accepted < 0)
                    or np.any(accepted > attempted) or int(attempted.sum()) != step):
                raise ValueError("conditional PIMD checkpoint counters are inconsistent")
            trial_rng = copy.deepcopy(self.rng)
            trial_rng.bit_generator.state = json.loads(str(saved["rng"]))
        self._q, self._internal = arrays["q"], arrays["internal"]
        self._orientations, self._centers = arrays["orientations"], arrays["centers"]
        self._energies = arrays["energies"]
        self._spring, self._log_weight = float(spring), float(weight)
        self._step, self._attempted, self._accepted = step, attempted.copy(), accepted.copy()
        self.rng.bit_generator.state = trial_rng.bit_generator.state
        return self
