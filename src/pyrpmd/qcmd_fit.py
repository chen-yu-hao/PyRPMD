"""Restartable intramolecular f-QCMD fitting by iterative Boltzmann inversion.

The reference is ordinary, unconstrained PIMD. Its bead-mean bond lengths
and angles are matched by classical NVT sampling on the base PES plus a
separable correction. This is the distribution-matching construction of
Fletcher et al., JCP 155, 231101 (2021), doi:10.1063/5.0076704, rather than
an estimator of bare holonomic constraint forces.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .checkpoint import atomic_savez, load_checkpoint
from .cmd import _integer, _real, _scalar
from .dynamics import Dynamics, State
from .io import fingerprint_payload
from .qcmd import (INTERPOLATION, QCMDModel, QuasiCentroidPotential,
                   _validate_base_potential)


FIT_KIND = "PyRPMD.fqcmd-fit.v1"
_PHASES = ("target_equilibration", "target_sampling", "ibi_equilibration",
           "ibi_sampling", "done")
_ARRAYS = {"q", "p", "target_bond", "target_angle", "sampled_bond",
           "sampled_angle", "correction_bond", "correction_angle", "metadata"}


def _probabilities(counts, *, regularization, smoothing):
    """Raw, smoothed, and regularized probabilities on a uniform grid.

    The same *probability* prior is used regardless of sampling length:
    ``p_reg=(1-epsilon)*p_smooth+epsilon/n_bins``. A count pseudocount would
    have unequal strengths for target and model trajectories of unequal
    lengths. The reflection boundary and Gaussian cutoff are explicit.
    """
    count = _real(counts, "histogram counts")
    if count.ndim != 1 or count.size == 0 or np.any(count < 0) or count.sum() <= 0:
        raise ValueError("histogram counts must be a nonnegative vector with positive total")
    raw = count / count.sum()
    smooth = (gaussian_filter1d(raw, smoothing, mode="reflect", truncate=4.)
              if smoothing else raw.copy())
    smooth /= smooth.sum()
    regular = (1. - regularization) * smooth + regularization / count.size
    if not np.isfinite(regular).all() or np.any(regular <= 0):
        raise ValueError("zero-density IBI bins require explicit positive regularization or more sampling")
    return raw, smooth, regular


def _ibi_update(correction, target, sampled, *, beta, damping, regularization,
                smoothing):
    """Conservative tabulated IBI increment, with an irrelevant fixed gauge."""
    target_probability = _probabilities(target, regularization=regularization,
                                        smoothing=smoothing)[2]
    sampled_probability = _probabilities(sampled, regularization=regularization,
                                         smoothing=smoothing)[2]
    update = damping / beta * (np.log(sampled_probability) - np.log(target_probability))
    updated = np.asarray(correction, dtype=float) + update
    # Keep the target-weighted energy zero; only derivatives affect dynamics.
    updated -= np.dot(target_probability, updated)
    if not np.isfinite(updated).all():
        raise FloatingPointError("nonfinite IBI correction")
    return updated


def _histograms(internal, grids, *, context):
    """Pool equivalent terminal bonds and angles, never discarding overflow."""
    values = (np.asarray(internal)[..., :2].ravel(), np.asarray(internal)[..., 2].ravel())
    result = []
    for name, sample, edges in zip(("bond", "angle"), values, grids):
        if not np.isfinite(sample).all():
            raise ValueError(f"nonfinite f-QCMD {name} in {context}")
        outside = (sample < edges[0]) | (sample > edges[-1])
        if np.any(outside):
            value = float(sample[np.flatnonzero(outside)[0]])
            raise ValueError(f"f-QCMD {name} value {value} outside grid "
                             f"[{edges[0]}, {edges[-1]}] in {context}; widen the fixed grid and refit")
        counts = np.histogram(sample, bins=edges)[0].astype(np.int64, copy=False)
        if int(counts.sum()) != sample.size:
            raise RuntimeError("f-QCMD histogram lost in-domain samples")
        result.append(counts)
    return result


def _diagnostic(iteration, target, sampled, correction, settings):
    entries = {}
    for name, reference, actual, value in zip(("bond", "angle"), target, sampled, correction):
        r = _probabilities(reference, regularization=settings["regularization"],
                           smoothing=settings["smoothing"])
        a = _probabilities(actual, regularization=settings["regularization"],
                           smoothing=settings["smoothing"])
        entries[name] = {
            "raw_total_variation": float(.5 * np.abs(r[0] - a[0]).sum()),
            "smoothed_total_variation": float(.5 * np.abs(r[1] - a[1]).sum()),
            "regularized_total_variation": float(.5 * np.abs(r[2] - a[2]).sum()),
            "raw_cdf_max_difference": float(np.max(np.abs(np.cumsum(r[0] - a[0])))),
            "target_counts": reference.tolist(), "sampled_counts": actual.tolist(),
            "target_count_total": int(reference.sum()), "sampled_count_total": int(actual.sum()),
            "target_empty_bins": int(np.count_nonzero(reference == 0)),
            "sampled_empty_bins": int(np.count_nonzero(actual == 0)),
            "target_edge_probability": float(r[0][0] + r[0][-1]),
            "sampled_edge_probability": float(a[0][0] + a[0][-1]),
            "correction_max_abs_hartree": float(np.max(np.abs(value))),
        }
    return {"iteration": iteration, "coordinates": entries,
            "evaluated_correction": "before_update", "converged": False,
            "physical_converged": False, "independent_sample_count": None}


def _read_fit(path, identity, settings, template):
    """Validate the complete saved state before constructing a force engine."""
    with load_checkpoint(path, expected_kind=FIT_KIND, require_integrity=True) as data:
        if set(data.files) != _ARRAYS:
            raise ValueError("invalid f-QCMD fitting checkpoint array inventory")
        metadata = json.loads(str(data["metadata"].item()))
        if not isinstance(metadata, dict) or set(metadata) != {
            "format", "identity", "settings", "phase", "phase_step", "iteration",
            "step", "time", "operators", "diagnostics",
        }:
            raise ValueError("invalid f-QCMD fitting checkpoint metadata")
        if (metadata["format"] != FIT_KIND or metadata["identity"] != identity
                or metadata["settings"] != settings):
            raise ValueError("f-QCMD fitting checkpoint scientific identity changed")
        phase = metadata["phase"]
        if phase not in _PHASES:
            raise ValueError("invalid f-QCMD fitting checkpoint phase")
        phase_step = _integer(metadata["phase_step"], "checkpoint phase step", 0)
        iteration = _integer(metadata["iteration"], "checkpoint iteration", -1)
        target_phase = phase.startswith("target_")
        if (target_phase and iteration != -1) or (not target_phase and not 0 <= iteration <= settings["iterations"]):
            raise ValueError("invalid f-QCMD fitting checkpoint iteration")
        q, p = data["q"], data["p"]
        expected_shape = (template.training_beads if target_phase else 1, len(template.masses), 3)
        if (q.dtype != np.float64 or p.dtype != np.float64
                or q.shape != expected_shape or p.shape != expected_shape):
            raise ValueError("invalid f-QCMD fitting checkpoint position/momentum shape or dtype")
        state = State(q, p, metadata["step"], metadata["time"])
        target = [data["target_bond"], data["target_angle"]]
        sampled = [data["sampled_bond"], data["sampled_angle"]]
        correction = [data["correction_bond"], data["correction_angle"]]
    for actual, expected in zip(target + sampled, list(template.corrections) * 2):
        if (actual.dtype.kind not in "iu" or actual.shape != expected.shape or np.any(actual < 0)):
            raise ValueError("invalid f-QCMD fitting checkpoint histogram counts")
    for actual, expected in zip(correction, template.corrections):
        if actual.dtype != np.float64 or actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError("invalid f-QCMD fitting checkpoint correction")
    limits = {"target_equilibration": settings["equilibration_steps"],
              "target_sampling": settings["sampling_steps"],
              "ibi_equilibration": settings["ibi_equilibration_steps"],
              "ibi_sampling": settings["ibi_sampling_steps"], "done": 0}
    if phase_step > limits[phase]:
        raise ValueError("invalid f-QCMD fitting checkpoint phase clock")
    interval = settings["sample_interval"]
    target_frames = (0 if phase == "target_equilibration" else
                     phase_step // interval if phase == "target_sampling" else
                     settings["sampling_steps"] // interval)
    sampled_frames = (phase_step // interval if phase == "ibi_sampling" else
                      settings["ibi_sampling_steps"] // interval if phase == "done" else 0)
    molecules = len(template.molecules)
    for counts, frames in ((target, target_frames), (sampled, sampled_frames)):
        if [int(c.sum()) for c in counts] != [frames * molecules * 2, frames * molecules]:
            raise ValueError("f-QCMD fitting checkpoint histogram/clock mismatch")
    target_steps = (phase_step if phase == "target_equilibration" else
                    settings["equilibration_steps"] + phase_step if phase == "target_sampling" else
                    settings["equilibration_steps"] + settings["sampling_steps"])
    ibi_steps = 0
    if not target_phase:
        cycle = settings["ibi_equilibration_steps"] + settings["ibi_sampling_steps"]
        ibi_steps = ((settings["iterations"] + 1) * cycle if phase == "done" else
                     iteration * cycle + phase_step +
                     (settings["ibi_equilibration_steps"] if phase == "ibi_sampling" else 0))
    expected_step = target_steps + ibi_steps
    expected_time = target_steps * settings["dt"] + ibi_steps * settings["ibi_dt"]
    if (state.step != expected_step or state.time < 0
            or not np.isclose(state.time, expected_time, rtol=2e-10, atol=1e-12)):
        raise ValueError("f-QCMD fitting checkpoint integration clock mismatch")
    diagnostics = metadata["diagnostics"]
    expected_diagnostics = 0 if target_phase else iteration + (phase == "done")
    if (not isinstance(diagnostics, list) or len(diagnostics) != expected_diagnostics
            or any(not isinstance(d, dict) or d.get("iteration") != i for i, d in enumerate(diagnostics))):
        raise ValueError("invalid f-QCMD fitting checkpoint diagnostics")
    if target_phase and any(np.any(c != 0) for c in correction):
        raise ValueError("target PIMD checkpoint cannot contain an effective correction")
    return metadata, state, target, sampled, correction


def fit_fqcmd(potential, masses, beta, initial_q, molecules, *, grids, dt,
              equilibration_steps, sampling_steps, sample_interval, iterations,
              ibi_equilibration_steps, ibi_sampling_steps, gamma_centroid=.01,
              damping=.5, regularization=.01, smoothing=1., seed=0,
              base_identity, checkpoint=None, checkpoint_interval=1000,
              ibi_dt=None):
    """Fit an equivalent-triatomic f-QCMD correction with exact continuation.

    All dimensional inputs use atomic units. ``initial_q`` is ``[P,N,3]``;
    grids are two fixed, uniform edge vectors for bonds (bohr) and angles
    (radian). All molecules and both terminal bonds are pooled; chemical
    equivalence is an explicit caller assumption. The model enforces matching
    masses and complete, nonoverlapping molecular coverage. No cell, frozen
    atoms, bond constraints, or alternative thermostat is accepted here.

    Standard normal-mode BAOAB/PILE-L supplies the reference histograms.
    Classical BAOAB at the same physical temperature then performs
    ``iterations`` IBI updates plus one final validation cycle. Every cycle
    starts with its own equilibration, and the returned diagnostics describe
    the actual returned potential. A single bead is permitted for classical
    limit checks. This driver does not assert physical convergence.

    ``regularization`` is a uniform *probability-mass* prior in ``[0,1)``;
    ``smoothing`` is a Gaussian width in bins (reflection boundary). With no
    positive prior, any remaining zero-density bin raises explicitly.
    Domain overflow also raises, rather than dropping counts or extrapolating.

    If ``checkpoint`` exists, all scientific settings must match before any
    PES call. Checkpoints contain counts, corrections, phase/clock, positions,
    momenta, and complete operator/RNG state in the common atomic container.
    Storage path and checkpoint interval do not affect the scientific identity.
    """
    _validate_base_potential(potential)
    q0 = _real(initial_q, "initial_q")
    if q0.ndim != 3 or q0.shape[-1] != 3 or min(q0.shape) == 0:
        raise ValueError("initial_q must have a nonempty [bead,atom,3] shape")
    if not isinstance(grids, (tuple, list)) or len(grids) != 2:
        raise ValueError("grids must contain bond and angle edge arrays")
    edges = [_real(e, "grid edges") for e in grids]
    if any(e.ndim != 1 or e.size < 5 for e in edges):
        raise ValueError("each f-QCMD grid requires at least four uniform bins")
    template = QCMDModel(molecules, masses, beta, edges,
                         [np.zeros(e.size - 1) for e in edges],
                         base_identity=base_identity, training_beads=q0.shape[0])
    if q0.shape[1] != len(template.masses):
        raise ValueError("initial_q atom count differs from masses")
    dt = _scalar(dt, "dt")
    ibi_dt = dt if ibi_dt is None else _scalar(ibi_dt, "ibi_dt")
    damping = _scalar(damping, "damping")
    regularization = _scalar(regularization, "regularization", positive=False)
    smoothing = _scalar(smoothing, "smoothing", positive=False)
    gamma_centroid = _scalar(gamma_centroid, "gamma_centroid")
    if damping > 1 or regularization >= 1:
        raise ValueError("damping must be in (0,1] and regularization in [0,1)")
    seed = _integer(seed, "seed", 0)
    equilibration_steps = _integer(equilibration_steps, "equilibration_steps", 0)
    sampling_steps = _integer(sampling_steps, "sampling_steps")
    sample_interval = _integer(sample_interval, "sample_interval")
    iterations = _integer(iterations, "iterations", 0)
    ibi_equilibration_steps = _integer(ibi_equilibration_steps, "ibi_equilibration_steps", 0)
    ibi_sampling_steps = _integer(ibi_sampling_steps, "ibi_sampling_steps")
    checkpoint_interval = _integer(checkpoint_interval, "checkpoint_interval")
    if sample_interval > min(sampling_steps, ibi_sampling_steps):
        raise ValueError("sampling_steps and ibi_sampling_steps must each include a sampling interval")
    geometry = template.coordinates
    _histograms(geometry.mean_values(q0), template.grids, context="initial geometry")
    settings = {
        "method": "f-qcmd", "algorithm": "unconstrained-pimd-ibi-v1",
        "base_identity": base_identity, "masses": template.masses.tolist(),
        "molecules": template.molecules.tolist(), "beta": template.beta,
        "training_beads": template.training_beads,
        "initial_q_identity": fingerprint_payload({"q": q0}),
        "grids": [e.tolist() for e in template.grids], "dt": dt, "ibi_dt": ibi_dt,
        "equilibration_steps": equilibration_steps, "sampling_steps": sampling_steps,
        "sample_interval": sample_interval, "iterations": iterations,
        "ibi_equilibration_steps": ibi_equilibration_steps,
        "ibi_sampling_steps": ibi_sampling_steps, "gamma_centroid": gamma_centroid,
        "damping": damping, "regularization": regularization, "smoothing": smoothing,
        "seed": seed, "integrator": "normal-mode-bab", "sampling_integrator": "baoab",
        "thermostat": "pile-l", "internal_friction": "2*omega_k",
        "regularization_rule": "(1-epsilon)*smoothed_probability+epsilon/n_bins",
        "smoothing_rule": "gaussian-bin-units;mode=reflect;truncate=4;renormalize",
        "correction_gauge": "regularized-target-probability-weighted mean zero",
        "pooling": "all equivalent triatomics; both equivalent terminal bonds",
        "interpolation": INTERPOLATION, "domain_policy": "error",
        "implementation": "numpy-float64", "intermolecular_corrections": False,
    }
    identity = fingerprint_payload(settings)
    path = None if checkpoint is None else Path(checkpoint)

    def make_model(correction, target, diagnostics):
        return QCMDModel(template.molecules, template.masses, template.beta,
                         template.grids, correction, base_identity=base_identity,
                         training_beads=template.training_beads,
                         target_counts=target, diagnostics=diagnostics,
                         training={**settings, "fit_identity": identity,
                                   "converged": False, "physical_converged": False})

    def make_dynamics(phase, iteration, correction, target, diagnostics, rng):
        reference = phase.startswith("target_")
        pes = potential if reference else QuasiCentroidPotential(
            potential, make_model(correction, target, diagnostics))
        return Dynamics(pes, template.masses, template.beta,
                        template.training_beads if reference else 1,
                        dt if reference else ibi_dt, seed=rng, native=False,
                        loop_backend="python", integrator="normal-mode-bab",
                        potential_identity=identity + (":target" if reference else f":ibi:{iteration}"))

    if path is not None and path.exists():
        metadata, state, target, sampled, correction = _read_fit(path, identity, settings, template)
        phase, phase_step = metadata["phase"], metadata["phase_step"]
        iteration, diagnostics = metadata["iteration"], metadata["diagnostics"]
        dynamics = make_dynamics(phase, iteration, correction, target, diagnostics, seed)
        dynamics.load_operator_state_dict(metadata["operators"])
        _histograms(geometry.mean_values(state.q), template.grids, context="checkpoint geometry")
    else:
        phase, phase_step, iteration, diagnostics = "target_equilibration", 0, -1, []
        target = [np.zeros_like(c, dtype=np.int64) for c in template.corrections]
        sampled = [c.copy() for c in target]
        correction = [c.copy() for c in template.corrections]
        dynamics = make_dynamics(phase, iteration, correction, target, diagnostics, seed)
        state = State(q0, dynamics.sample_momenta())

    def save():
        if path is None:
            return
        metadata = {"format": FIT_KIND, "identity": identity, "settings": settings,
                    "phase": phase, "phase_step": phase_step, "iteration": iteration,
                    "step": state.step, "time": state.time,
                    "operators": dynamics.operator_state_dict(), "diagnostics": diagnostics}
        atomic_savez(path, kind=FIT_KIND,
                     metadata=json.dumps(metadata, sort_keys=True, allow_nan=False),
                     q=state.q, p=state.p, target_bond=target[0], target_angle=target[1],
                     sampled_bond=sampled[0], sampled_angle=sampled[1],
                     correction_bond=correction[0], correction_angle=correction[1])

    save()
    while phase != "done":
        is_target = phase.startswith("target_")
        is_sampling = phase.endswith("_sampling")
        limit = (sampling_steps if is_sampling else equilibration_steps) if is_target else (
            ibi_sampling_steps if is_sampling else ibi_equilibration_steps)
        while phase_step < limit:
            # Observation failure is transactional too: a domain error must
            # not leave a committed integration step with missing counts.
            observing = is_sampling and (phase_step + 1) % sample_interval == 0
            previous = state.copy() if observing else None
            previous_rng = copy.deepcopy(dynamics.rng.bit_generator.state) if observing else None
            try:
                dynamics.sample_step(state, gamma_centroid=gamma_centroid,
                                     thermostat="pile-l", method="baoab")
                if observing:
                    internal = geometry.mean_values(state.q) if is_target else geometry.values(state.q)[0]
                    counts = _histograms(internal, template.grids,
                                         context=f"{phase}, iteration {iteration}, step {phase_step + 1}")
            except BaseException:
                if previous is not None:
                    state.q[:], state.p[:] = previous.q, previous.p
                    state.step, state.time = previous.step, previous.time
                    dynamics.rng.bit_generator.state = previous_rng
                raise
            phase_step += 1
            if observing:
                collector = target if is_target else sampled
                for cumulative, increment in zip(collector, counts):
                    cumulative += increment
            if state.step % checkpoint_interval == 0:
                save()
        if phase == "target_equilibration":
            phase, phase_step = "target_sampling", 0
        elif phase == "target_sampling":
            classical_q = geometry.quasicentroids(state.q, template.masses)[None]
            phase, phase_step, iteration = "ibi_equilibration", 0, 0
            dynamics = make_dynamics(phase, iteration, correction, target, diagnostics, dynamics.rng)
            state = State(classical_q, dynamics.sample_momenta(), state.step, state.time)
        elif phase == "ibi_equilibration":
            phase, phase_step = "ibi_sampling", 0
        else:
            diagnostics.append(_diagnostic(iteration, target, sampled, correction, settings))
            if iteration == iterations:
                phase, phase_step = "done", 0
            else:
                correction = [_ibi_update(c, t, s, beta=template.beta, damping=damping,
                                           regularization=regularization, smoothing=smoothing)
                              for c, t, s in zip(correction, target, sampled)]
                sampled = [np.zeros_like(c) for c in target]
                iteration += 1
                phase, phase_step = "ibi_equilibration", 0
                dynamics = make_dynamics(phase, iteration, correction, target, diagnostics, dynamics.rng)
        save()
    return make_model(correction, target, diagnostics)
