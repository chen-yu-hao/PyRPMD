"""Restrained-ensemble string sampling in nonperiodic centroid CVs.

The existing :mod:`adaptive_string` owns string geometry.  This module adds
thermal sampling and the mass metric to that geometry.  At finite stiffness K,
``K @ (z - <xi>)`` is the gradient of the *restrained*, Gaussian-smoothed free
energy.  It approaches the marginal-CV free-energy gradient as K increases;
neither finite stiffness nor finite sampling is silently declared converged.

The metric is <G M^-1 G^T>, with G = sum_p jacobian[:,p,:,:].  Thus a centroid
CV's 1/P Jacobian is undone exactly once.  Restraining the physical centroid
in the full Cartesian ensemble includes its Jacobian measure automatically;
no extra radial/Fixman potential is added to the umbrella Hamiltonian.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .adaptive_string import _real_array, _positive_real, _positive_integer, _validated, _reparameterize
from .analysis.fes_statistics import cv_statistical_inefficiency
from .bias import UmbrellaBias
from .checkpoint import atomic_savez, load_checkpoint
from .dynamics import Dynamics, State
from .engine import _json
from .validation import checked_integer, finite_real


def _centroid_cv(cv):
    components = getattr(cv, "cvs", None)
    centroid = (all(_centroid_cv(item) for item in components) if components is not None
                else getattr(cv, "estimator", None) == "centroid")
    if not centroid or any(period is not None for period in cv.periods):
        raise ValueError("finite-temperature string requires nonperiodic centroid CVs")
    return True


def _metric(jacobian, masses):
    gradient = jacobian.sum(axis=1)
    return np.einsum("iac,jac,a->ij", gradient, gradient, 1.0 / masses)


@dataclass(frozen=True)
class StringStatistics:
    """One complete restrained sampling batch, one row per string image.

    Covariances describe errors of estimated means from contiguous blocks.
    They do not include finite-restraint error or failure to equilibrate.
    ``drift_covariance`` includes mean-force/metric covariance within blocks.
    """

    gradient: np.ndarray
    metric: np.ndarray
    gradient_covariance: np.ndarray
    drift_covariance: np.ndarray
    effective_samples: np.ndarray
    blocks: int
    correlation_resolved: bool

    def validate(self, shape):
        count, dimension = shape
        for name in ("gradient", "metric", "gradient_covariance", "drift_covariance"):
            array = _real_array(getattr(self, name), name)
            expected = shape if name == "gradient" else (count, dimension, dimension)
            if array.shape != expected:
                raise ValueError(f"string {name} has incorrect shape")
            if name != "gradient":
                if not np.allclose(array, array.swapaxes(-1, -2), rtol=1e-12, atol=1e-14):
                    raise ValueError(f"string {name} must be symmetric")
                eigenvalues = np.linalg.eigvalsh(array)
                if name == "metric" and np.any(eigenvalues[..., 0] <= 1e-12 * eigenvalues[..., -1]):
                    raise ValueError("string CV metric must be positive definite and nonsingular")
                if name != "metric" and np.any(eigenvalues[..., 0] < -1e-12 * np.maximum(1., eigenvalues[..., -1])):
                    raise ValueError(f"string {name} must be positive semidefinite")
        ess = _real_array(self.effective_samples, "effective_samples")
        if ess.shape != (count,) or np.any(ess <= 0):
            raise ValueError("string effective sample counts must be positive per image")
        _positive_integer(self.blocks, "blocks")
        if not isinstance(self.correlation_resolved, (bool, np.bool_)):
            raise ValueError("correlation_resolved must be boolean")


def restrained_string_statistics(images, values, metrics, stiffness, *, block_size=10):
    """Reduce thermal window observations without assuming independent frames.

    ``values[M,S,D]`` and instantaneous ``metrics[M,S,D,D]`` must come from
    equilibrated independent restrained windows centered at ``images[M,D]``.
    Samples within a window are time ordered.  A multiple of ``block_size`` is
    required, avoiding a different sample prefix in the mean and its error.
    This function also supports externally generated exact reference samples.
    """
    images = _validated(images)
    values = _real_array(values, "sampled CV values")
    metrics = _real_array(metrics, "sampled CV metrics")
    count, dimension = images.shape
    if values.ndim != 3 or values.shape[0] != count or values.shape[2] != dimension:
        raise ValueError("sampled CV values must have shape [images,samples,dimension]")
    samples = values.shape[1]
    if metrics.shape != (count, samples, dimension, dimension):
        raise ValueError("sampled CV metrics must have shape [images,samples,dimension,dimension]")
    eigenvalues = np.linalg.eigvalsh(metrics)
    if (not np.allclose(metrics, metrics.swapaxes(-1, -2), rtol=1e-12, atol=1e-14)
            or np.any(eigenvalues[..., 0] <= 1e-12 * eigenvalues[..., -1])):
        raise ValueError("each sampled CV metric must be symmetric positive definite and nonsingular")
    block_size = _positive_integer(block_size, "block_size")
    blocks = samples // block_size
    if blocks < 2 or samples % block_size:
        raise ValueError("sampling needs at least two complete blocks with no remainder")
    k = _real_array(stiffness, "stiffness")
    if k.ndim == 0:
        k = np.eye(dimension) * k
    elif k.ndim == 1:
        k = np.diag(k)
    if (k.shape != (dimension, dimension) or not np.allclose(k, k.T)
            or np.linalg.eigvalsh(k).min() <= 0):
        raise ValueError("string stiffness must be symmetric positive definite")
    gradients = np.einsum("ij,msj->msi", k, images[:, None, :] - values)
    block_gradients = gradients.reshape(count, blocks, block_size, dimension).mean(axis=2)
    block_metrics = metrics.reshape(count, blocks, block_size, dimension, dimension).mean(axis=2)
    block_drifts = np.einsum("mbij,mbj->mbi", block_metrics, block_gradients)

    def covariance(blocks_array):
        centered = blocks_array - blocks_array.mean(axis=1, keepdims=True)
        return np.einsum("mbi,mbj->mij", centered, centered) / (blocks * (blocks - 1))

    inefficiency = np.array([cv_statistical_inefficiency(value) for value in values])
    result = StringStatistics(
        gradients.mean(axis=1), metrics.mean(axis=1), covariance(block_gradients),
        covariance(block_drifts), samples / inefficiency, blocks,
        bool(blocks >= 20 and np.all(block_size >= 5 * inefficiency)),
    )
    result.validate(images.shape)
    return result


class RestrainedStringSampler:
    """Sample one independent PILE-L trajectory at each fixed string image.

    ``dynamics`` is a sequence of independent ordinary ``Dynamics`` instances;
    states are copied and retained between string iterations.  Coordinates,
    masses, inverse temperature, timestep, friction and K are in atomic units.
    Each call re-equilibrates after moving the restraints, then records a fixed
    number of samples.  Nonperiodic CVs are required; atom constraints, pressure
    coupling and force splitting need a separately derived metric and fail.
    """

    def __init__(self, dynamics, cv, states, *, stiffness, equilibration_steps=100,
                 sampling_steps=1000, sample_interval=10, block_size=10,
                 gamma_centroid=1., cv_identity=None):
        self.dynamics = tuple(dynamics)
        self.states = [state.copy() for state in states]
        self.cv = cv
        _centroid_cv(cv)
        if cv.dimension < 2 or len(self.dynamics) < 3 or len(self.states) != len(self.dynamics):
            raise ValueError("string needs at least three image states and at least two CVs")
        if len({id(d) for d in self.dynamics}) != len(self.dynamics):
            raise ValueError("string images need independent Dynamics objects")
        if len({id(d.rng) for d in self.dynamics}) != len(self.dynamics):
            raise ValueError("string images must not share RNG objects")
        streams = {json.dumps(_json(d.rng.bit_generator.state), sort_keys=True) for d in self.dynamics}
        if len(streams) != len(self.dynamics):
            raise ValueError("string image RNG streams must have distinct states; spawn or seed them independently")
        for dynamics, state in zip(self.dynamics, self.states):
            if (type(dynamics) is not Dynamics or dynamics.plan.active or dynamics.bond_constraints
                    or dynamics.fixed_atoms or dynamics.integrator_name == "velocity-verlet"):
                raise ValueError("string sampler needs unconfigured, unconstrained Cayley or normal-mode Dynamics")
            dynamics.integrator._check(state.q, state.p)
            if dynamics.dt <= 0:
                raise ValueError("string sampler timestep must be positive")
            first = self.dynamics[0]
            if (dynamics.beta != first.beta or dynamics.nbeads != first.nbeads
                    or not np.array_equal(dynamics.masses, first.masses)):
                raise ValueError("string images must share temperature, masses and bead count")
            if (dynamics.potential is not first.potential
                    and (dynamics.potential_identity is None
                         or dynamics.potential_identity != first.potential_identity)):
                raise ValueError("string images need a common PES object or a matching explicit potential_identity")
        if (isinstance(equilibration_steps, (bool, np.bool_))
                or not isinstance(equilibration_steps, (int, np.integer)) or equilibration_steps < 0):
            raise ValueError("equilibration_steps must be a nonnegative integer")
        self.equilibration_steps = int(equilibration_steps)
        self.sampling_steps = _positive_integer(sampling_steps, "sampling_steps")
        self.sample_interval = _positive_integer(sample_interval, "sample_interval")
        self.block_size = _positive_integer(block_size, "block_size")
        samples = self.sampling_steps // self.sample_interval
        if self.sampling_steps % self.sample_interval or samples % self.block_size or samples < 2 * self.block_size:
            raise ValueError("sampling_steps must produce at least two complete blocks without remainder")
        self.gamma_centroid = _positive_real(gamma_centroid, "gamma_centroid")
        self.stiffness = UmbrellaBias(cv, np.zeros(cv.dimension), stiffness).stiffness.copy()
        self.stiffness.flags.writeable = False
        if cv_identity is not None and (not isinstance(cv_identity, str) or not cv_identity.strip()):
            raise ValueError("cv_identity must be a nonempty scientific identity")
        self.cv_identity = cv_identity
        self._initial_clocks = [(state.step, state.time) for state in self.states]
        self._batches = 0

    def __call__(self, images):
        images = _validated(images)
        if images.shape != (len(self.states), self.cv.dimension):
            raise ValueError("string images must match sampler states and CV dimensions")
        previous = self.snapshot()
        values, metrics = [], []
        try:
            for image, dynamics, state in zip(images, self.dynamics, self.states):
                bias = UmbrellaBias(self.cv, image, self.stiffness)
                window_values, window_metrics = [], []
                for index in range(self.equilibration_steps + self.sampling_steps):
                    dynamics.sample_step(state, bias, gamma_centroid=self.gamma_centroid)
                    production = index + 1 - self.equilibration_steps
                    if production > 0 and production % self.sample_interval == 0:
                        result = self.cv.value_and_jacobian(state.q)
                        value = _real_array(result.value, "CV value")
                        jac = _real_array(result.jacobian, "CV Jacobian")
                        if value.shape != (self.cv.dimension,) or jac.shape != (self.cv.dimension,) + state.q.shape:
                            raise ValueError("string CV violates value[D]/Jacobian[D,P,N,3] contract")
                        # A centroid declaration must agree with its bead derivatives.
                        if not np.allclose(jac, jac[:, :1], rtol=1e-10, atol=1e-13):
                            raise ValueError("centroid string CV has bead-dependent derivatives")
                        window_values.append(value)
                        window_metrics.append(_metric(jac, dynamics.masses))
                values.append(window_values)
                metrics.append(window_metrics)
            statistics = restrained_string_statistics(images, values, metrics, self.stiffness,
                                                       block_size=self.block_size)
            _perpendicular_drift(images, statistics)
            self._batches += 1
            return statistics
        except BaseException:
            self.restore(previous)
            raise

    def identity(self):
        """Scientific settings; stable PES and custom-CV identities are explicit."""
        from .workflows.sampling import _checkpoint_cv_identity

        cv = self.cv_identity if self.cv_identity is not None else _checkpoint_cv_identity(self.cv)
        physics = []
        for dynamics in self.dynamics:
            if dynamics.potential_identity is None:
                raise ValueError("string checkpoint requires Dynamics.potential_identity for every image")
            physics.append({"potential": dynamics.potential_identity, "masses": dynamics.masses,
                            "beta": dynamics.beta, "beads": dynamics.nbeads, "dt": dynamics.dt,
                            "integrator": dynamics.integrator_name, "cell": dynamics.cell,
                            "pbc": dynamics.pbc})
        return _json({"cv": cv, "cv_units": self.cv.units, "cv_periods": self.cv.periods,
                      "cv_dimension": self.cv.dimension, "physics": physics, "stiffness": self.stiffness,
                      "equilibration_steps": self.equilibration_steps,
                      "sampling_steps": self.sampling_steps, "sample_interval": self.sample_interval,
                      "block_size": self.block_size, "gamma_centroid": self.gamma_centroid})

    def snapshot(self):
        return {"states": [dict(q=s.q.tolist(), p=s.p.tolist(), step=s.step, time=s.time)
                           for s in self.states],
                "operators": [d.operator_state_dict() for d in self.dynamics],
                "initial_clocks": [list(clock) for clock in self._initial_clocks],
                "batches": self._batches}

    def restore(self, snapshot):
        if not isinstance(snapshot, dict) or set(snapshot) != {"states", "operators", "initial_clocks", "batches"}:
            raise ValueError("invalid string sampler checkpoint")
        if (len(snapshot["states"]) != len(self.states)
                or len(snapshot["operators"]) != len(self.dynamics)
                or len(snapshot["initial_clocks"]) != len(self.states)):
            raise ValueError("string sampler checkpoint image count mismatch")
        batches = checked_integer(snapshot["batches"], "string checkpoint batches", minimum=0)
        candidates = [State(**value) for value in snapshot["states"]]
        clocks = []
        for dynamics, state, clock in zip(self.dynamics, candidates, snapshot["initial_clocks"]):
            dynamics.integrator._check(state.q, state.p)
            if not isinstance(clock, (list, tuple)) or len(clock) != 2:
                raise ValueError("invalid string initial clock")
            first_step = checked_integer(clock[0], "string initial step", minimum=0)
            first_time = finite_real(clock[1], "string initial time")
            elapsed = batches * (self.equilibration_steps + self.sampling_steps)
            if (state.step != first_step + elapsed
                    or not np.isclose(state.time, first_time + elapsed * dynamics.dt, rtol=1e-11, atol=1e-11)):
                raise ValueError("string checkpoint clock disagrees with complete sampling batches")
            clocks.append((first_step, first_time))
        old_operators = [d.operator_state_dict() for d in self.dynamics]
        try:
            for dynamics, operators in zip(self.dynamics, snapshot["operators"]):
                dynamics.load_operator_state_dict(operators)
        except BaseException:
            for dynamics, operators in zip(self.dynamics, old_operators):
                dynamics.load_operator_state_dict(operators)
            raise
        self.states = candidates
        self._initial_clocks = clocks
        self._batches = batches


@dataclass(frozen=True)
class FiniteTemperatureStringResult:
    images: np.ndarray
    statistics: StringStatistics
    free_energy: np.ndarray
    free_energy_standard_error: np.ndarray
    maximum_perpendicular_drift: float
    maximum_perpendicular_standard_error: float
    converged: bool
    iterations: int
    history: tuple
    diagnostics: dict


def _perpendicular_drift(path, statistics):
    if not isinstance(statistics, StringStatistics):
        raise TypeError("string sampler must return StringStatistics")
    statistics.validate(path.shape)
    tangent = path[2:] - path[:-2]
    norm = np.linalg.norm(tangent, axis=1)
    if np.any(norm <= 0):
        raise ValueError("undefined string tangent")
    tangent /= norm[:, None]
    projection = np.eye(path.shape[1])[None] - np.einsum("mi,mj->mij", tangent, tangent)
    drift = np.einsum("mij,mj->mi", statistics.metric, statistics.gradient)
    force = np.einsum("mij,mj->mi", projection, drift[1:-1])
    covariance = np.einsum("mij,mjk,mkl->mil", projection,
                           statistics.drift_covariance[1:-1], projection)
    errors = np.sqrt(np.maximum(np.trace(covariance, axis1=1, axis2=2), 0.))
    if not np.isfinite(force).all() or not np.isfinite(errors).all():
        raise ValueError("nonfinite metric-weighted string drift or error")
    return force, float(np.linalg.norm(force, axis=1).max()), float(errors.max())


def _path_integral(path, statistics):
    segments = np.diff(path, axis=0)
    count, dimension = path.shape
    weights = np.zeros((count, count, dimension))
    for end in range(1, count):
        weights[end] = weights[end - 1]
        weights[end, end - 1:end + 1] += 0.5 * segments[end - 1]
    free_energy = np.einsum("mij,ij->m", weights, statistics.gradient)
    variance = np.einsum("mij,ijk,mik->m", weights, statistics.gradient_covariance, weights)
    return free_energy, np.sqrt(np.maximum(variance, 0.))


def finite_temperature_string(images, sampler, *, step=0.01, tolerance=1e-3,
                              max_iterations=100, confidence_z=2., checkpoint=None,
                              resume=False):
    """Relax a fixed-endpoint CV string using sampled metric-weighted forces.

    ``sampler(images)`` returns :class:`StringStatistics`; the bundled sampler
    performs restrained RPMD and supports exact restarts at iteration boundaries.
    Custom thermal samplers may be used without checkpointing.  ``max_iterations``
    is a total budget, so increasing it on resume extends the same random streams.
    The geometric stopping criterion includes ``confidence_z`` block errors;
    Independent stiffness, time-step, equilibration, image-spacing and sampling
    studies are required before treating the result as physically validated.
    """
    path = _validated(images)
    initial = path.copy()
    step = _positive_real(step, "step")
    tolerance = _positive_real(tolerance, "tolerance")
    max_iterations = _positive_integer(max_iterations, "max_iterations")
    confidence_z = _positive_real(confidence_z, "confidence_z")
    if not callable(sampler):
        raise TypeError("string sampler must be callable")
    if not isinstance(resume, bool) or resume and checkpoint is None:
        raise ValueError("resume must be boolean and requires a checkpoint path")
    identity = None
    if checkpoint is not None:
        if type(sampler) is not RestrainedStringSampler:
            raise ValueError("checkpointing requires the bundled RestrainedStringSampler")
        if Path(checkpoint).exists() and not resume:
            raise ValueError("string checkpoint already exists; select resume=True or a new path")
        identity = _json({"initial_images": initial, "sampler": sampler.identity(),
                          "step": step, "tolerance": tolerance, "confidence_z": confidence_z})
    iteration, history = 0, []
    if resume:
        with load_checkpoint(checkpoint, expected_kind="PyRPMD.thermal-string.v1", require_integrity=True) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if metadata.get("identity") != identity:
                raise ValueError("string checkpoint scientific identity differs")
            path = _validated(data["images"])
            if path.shape != initial.shape or not np.array_equal(path[[0, -1]], initial[[0, -1]]):
                raise ValueError("string checkpoint endpoints or dimensions differ")
            iteration = metadata["iteration"]
            if type(iteration) is not int or not 0 <= iteration <= max_iterations:
                raise ValueError("invalid string checkpoint iteration or reduced budget")
            history = _real_array(data["history"], "string history").tolist()
            if len(history) != iteration + 1:
                raise ValueError("string checkpoint history disagrees with iteration")
            statistics = StringStatistics(**metadata["statistics"])
            statistics = StringStatistics(*(np.asarray(getattr(statistics, key)) for key in (
                "gradient", "metric", "gradient_covariance", "drift_covariance", "effective_samples")),
                statistics.blocks, statistics.correlation_resolved)
            statistics.validate(path.shape)
            sampler_state = metadata["sampler_state"]
    else:
        statistics = sampler(path.copy())
    force, residual, error = _perpendicular_drift(path, statistics)
    if resume and not np.isclose(history[-1], residual, rtol=0, atol=1e-13):
        raise ValueError("string checkpoint residual does not describe saved images")
    if resume:
        sampler.restore(sampler_state)
    if not resume:
        history.append(residual)

    def save():
        if checkpoint is not None:
            metadata = _json({"format": "PyRPMD.thermal-string.v1", "identity": identity,
                              "iteration": iteration, "statistics": vars(statistics),
                              "sampler_state": sampler.snapshot()})
            atomic_savez(checkpoint, kind="PyRPMD.thermal-string.v1", images=path,
                         history=np.asarray(history), metadata=np.asarray(json.dumps(metadata, allow_nan=False)))

    save()
    while residual + confidence_z * error > tolerance and iteration < max_iterations:
        moved = path.copy()
        moved[1:-1] -= step * force
        candidate = _reparameterize(moved, len(moved))
        next_statistics = sampler(candidate.copy())
        next_force, next_residual, next_error = _perpendicular_drift(candidate, next_statistics)
        path, statistics = candidate, next_statistics
        force, residual, error = next_force, next_residual, next_error
        iteration += 1
        history.append(residual)
        save()
    free_energy, free_energy_error = _path_integral(path, statistics)
    return FiniteTemperatureStringResult(
        path.copy(), statistics, free_energy, free_energy_error, residual, error,
        bool(residual + confidence_z * error <= tolerance), False, iteration, tuple(history),
        {"measure": "finite-restraint smoothed centroid free energy",
         "metric": "mean mass-inverse centroid Jacobian product",
         "uncertainty": "contiguous-block errors; no finite-restraint or discretization error",
         "correlation_resolved": statistics.correlation_resolved,
         "physical_convergence_requires": ["restraint-stiffness study", "time-step study",
            "equilibration and mixing checks", "image-spacing study", "independent replicas"]},
    )
