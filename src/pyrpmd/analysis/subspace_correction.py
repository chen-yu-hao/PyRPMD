"""Subspace quantum correction of full-dimensional RPMD rate constants.

    k ~= k_RPMD(full system) * C_S,      C_S = k_QM(H_S) / k_RPMD(H_S)

``H_S`` is a one- to three-dimensional subspace Hamiltonian
(:mod:`pyrpmd.analysis.subspace_model`).  The exact quantum rate
(:mod:`pyrpmd.analysis.subspace_quantum`) and the RPMD rate
(:mod:`pyrpmd.analysis.subspace_rpmd`) are computed for the *same* ``H_S``
and the same reactant normalization, so ``C_S`` isolates the RPMD method
error inside ``S`` -- deep tunnelling, resonances, coherence -- while the
full RPMD rate keeps everything RPMD already describes for the environment
(zero-point energy, adiabatic following, recrossing).

The correction is multiplicative, not additive: rates of the subspace model
and of the full system may differ by orders of magnitude (the environment
changes zero-point energies and partition functions), but a common
prefactor cancels in the ratio.  ``C_S`` is exact when ``S`` is the full
system and is a transferable-defect approximation otherwise; use
:func:`nested_subspace_sensitivity` (does ``C_S`` change when a coordinate
is added?) and the curvature-times-width test to decide whether ``S`` holds
the RPMD defect.  Nothing here marks a rate as physically converged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .subspace_model import SubspaceModel
from .subspace_quantum import QuantumRateResult, _json_safe, exact_quantum_rate
from .subspace_rpmd import RingPolymerRateResult, ring_polymer_rate

SCHEMA = "pyrpmd.subspace_correction"
CORRECTED_SCHEMA = "pyrpmd.rate_quantum_corrected"

# A gate that fails makes the correction invalid unless its policy is "allow"
# (an explicit, recorded override).  ``max_outside_fraction`` bounds the share
# of estimator samples that saw the constant continuation of a gridded V_S.
VALIDITY_POLICY = {"unconverged_path": "reject", "unstable_bath": "reject", "outside_grid": "reject",
                   "max_outside_fraction": 0.01, "submerged_surface": "reject", "max_submerged_kt": 1.0,
                   "folded_ribbon": "reject", "max_curvature_width": 0.5}


def validity_policy(policy: Mapping[str, Any] | None = None) -> dict:
    """Complete and check a validity policy (see :data:`VALIDITY_POLICY`)."""
    merged = dict(VALIDITY_POLICY)
    for key, value in dict(policy or {}).items():
        if key in merged and value is not None:
            merged[key] = value
    for key in ("unconverged_path", "unstable_bath", "outside_grid", "submerged_surface", "folded_ribbon"):
        if merged[key] not in ("reject", "allow"):
            raise ValueError(f"validity policy {key} must be 'reject' or 'allow'")
    limit = float(merged["max_outside_fraction"])
    if not np.isfinite(limit) or not 0 <= limit <= 1:
        raise ValueError("max_outside_fraction must lie in [0, 1]")
    merged["max_outside_fraction"] = limit
    depth = float(merged["max_submerged_kt"])
    if not np.isfinite(depth) or depth < 0:
        raise ValueError("max_submerged_kt must be finite and nonnegative")
    merged["max_submerged_kt"] = depth
    fold = float(merged["max_curvature_width"])
    if not np.isfinite(fold) or fold <= 0:
        raise ValueError("max_curvature_width must be finite and positive")
    merged["max_curvature_width"] = fold
    return merged


def _gate(name, passed, policy, reason, detail):
    passed = bool(passed)
    return {"gate": name, "passed": passed, "policy": policy,
            "overridden": bool(not passed and policy == "allow"),
            "reason": None if passed else reason, "detail": detail}


_PATH_EVIDENCE = ("status", "steps", "arc_length", "tail_energy_range", "asymptote_tolerance")
# Recorded by newer path builders only; older reports keep their five fields.
_PATH_TERMINUS = ("accepted", "reason", "end_energy", "drop_below_top", "required_drop", "gradient_norm")


def _path_evidence(end):
    return {k: end.get(k) for k in _PATH_EVIDENCE} | {k: end[k] for k in _PATH_TERMINUS if k in end}


def validity_gates(construction: Mapping[str, Any], rpmd_diagnostics: Mapping[str, Any] | None = None,
                   policy: Mapping[str, Any] | None = None) -> dict:
    """Decide whether a subspace model may carry a correction factor.

    ``construction`` is the model provenance.  Five gates are evaluated when
    their evidence is present:

    ``path``   both continuations of an assembled reaction path ended at an
               accepted terminus (``path_termination``, set by the pipeline):
               an asymptote, the reactant reference ``xi = 0`` after climbing
               out of a pre-reaction complex, or on the product side a well
               deep enough below the barrier to commit products; and the
               reactant reference slice of a 1D model lies in the free
               reactant channel (``reactant_reference``);
    ``bath``   the normal-plane Hessian of a path projection is stable
               (``bath_stable``);
    ``grid``   the ring-polymer estimator did not sample the constant
               continuation outside a gridded ``V_S`` (``grid_exit`` of the
               RPMD diagnostics; absent before the rates are solved);
    ``ribbon`` the path of an RPH 2D model bends less tightly than its
               transverse mode is wide: ``kappa * sigma`` at most
               ``max_curvature_width`` (``ribbon_fold``).  Beyond it the ribbon
               coordinates fold inside the transverse wavefunction;
    ``surface`` the RPMD dividing surface lies at most ``max_submerged_kt``
               below the centroid free-energy maximum between the reactant
               reference and the surface (``free_energy_bottleneck``).  A
               submerged barrier behind a complex fails it: QTST through it
               exceeds what the bottleneck allows, and the finite-time
               transmission coefficient does not remove the excess.

    The policy stored in the provenance (``validity_policy``) is used unless
    ``policy`` is given.  Returns ``{"valid", "failures", "overrides",
    "gates", "policy"}``; ``failures`` lists the gates that reject.
    """
    construction = dict(construction or {})
    policy = validity_policy(policy if policy is not None else construction.get("validity_policy"))
    gates = []
    termination = construction.get("path_termination")
    reference = construction.get("reactant_reference")
    if termination is not None:
        ends = {side: dict(termination.get(side) or {}) for side in ("reactant", "product")}
        bad = [f"{side}: {end.get('status', 'no termination record')}"
               + (f" -- {end['reason']}" if end.get("reason") else "")
               for side, end in ends.items() if not end.get("converged")]
        detail = {side: _path_evidence(end) for side, end in ends.items()}
        if reference is not None:
            detail["reactant_reference"] = dict(reference)
            if reference.get("rule") == "not_free":
                bad.append(f"reactant reference slice: V lies {reference['offset_kt']:.3g} kT from the "
                           f"reactant end (limit {reference['reference_kt']:.3g} kT) -- no free reactant channel")
        gates.append(_gate("path", not bad, policy["unconverged_path"],
                           "reaction path did not reach an accepted terminus (" + "; ".join(bad) + ")",
                           detail))
    if "bath_stable" in construction and construction.get("environment") != "frozen":
        unstable = list(construction.get("significant_unstable") or [])
        worst = min(unstable, key=lambda r: r["curvature"]) if unstable else None
        reason = None
        if worst is not None:
            reason = (f"unstable transverse Hessian at {len(unstable)} path points (worst "
                      f"{worst['imaginary_frequency_cm']:.0f}i cm^-1, "
                      f"{', '.join(construction.get('unstable_characters') or ['unclassified'])})")
        gates.append(_gate("bath", construction["bath_stable"], policy["unstable_bath"], reason, {
            "significant_unstable_points": len(unstable),
            "minimum_complement_curvature": construction.get("minimum_complement_curvature"),
            "unstable_characters": construction.get("unstable_characters"),
            "soft_complement_points": construction.get("soft_complement_points"),
            "dropped_zero_point_bound": construction.get("dropped_zero_point_bound"),
            "unstable_arc_lengths": [r["arc_length"] for r in unstable],
        }))
    fold = construction.get("ribbon_fold")
    if fold is not None and fold.get("max_curvature_width") is not None:
        value, limit = float(fold["max_curvature_width"]), policy["max_curvature_width"]
        gates.append(_gate("ribbon", value <= limit, policy["folded_ribbon"],
                           f"the RPH path bends with radius {fold['radius']:.3g} at arc length {fold['arc']:.6g}, "
                           f"inside the transverse width {fold['width']:.3g} (kappa * sigma = {value:.3g}, limit "
                           f"{limit:.3g}): the ribbon coordinates fold inside the wavefunction; use "
                           "model_2d = \"plane\"", dict(fold, max_curvature_width_limit=limit)))
    exits = (rpmd_diagnostics or {}).get("grid_exit")
    if exits is not None:
        limit = policy["max_outside_fraction"]
        worst = float(exits["estimator_outside_fraction"])
        # Which estimator saw the continuation decides the remedy: TI and
        # dividing-surface samples need a wider box across the channel, recrossing
        # trajectories that return or leave uphill need a box (or absorber) further along it.
        parts = {"ti": exits.get("ti_sample_outside_fraction"),
                 "surface": exits.get("surface_sample_outside_fraction"),
                 "recrossing": exits.get("recrossing_contaminated_flux_fraction")}
        known = {k: float(v) for k, v in parts.items() if v is not None}
        dominant = max(known, key=known.get) if known else None
        components = ", ".join(f"{k} {v:.3g}" for k, v in known.items())
        gates.append(_gate("grid", worst <= limit, policy["outside_grid"],
                           f"{worst:.3g} of the ring-polymer estimator sampled V_S outside its grid "
                           f"(limit {limit:.3g}; {components or 'no components'}; dominant: {dominant})",
                           {k: exits.get(k) for k in (
                               "ti_sample_outside_fraction", "surface_sample_outside_fraction",
                               "recrossing_exited_fraction", "recrossing_returned_fraction",
                               "recrossing_uphill_exit_fraction", "recrossing_contaminated_flux_fraction",
                               "recrossing_max_edge_slope", "estimator_outside_fraction")}
                           | {"max_outside_fraction": limit, "dominant_component": dominant}))
    bottleneck = (rpmd_diagnostics or {}).get("free_energy_bottleneck")
    if bottleneck is not None:
        depth = float(bottleneck["surface_below_maximum_kt"])
        limit = policy["max_submerged_kt"]
        gates.append(_gate("surface", depth <= limit, policy["submerged_surface"],
                           f"the RPMD dividing surface lies {depth:.3g} kT below the centroid free-energy "
                           f"maximum at axis-0 position {bottleneck['maximum_position']:.6g} (limit {limit:.3g} kT); "
                           "place the dividing surface at that bottleneck",
                           dict(bottleneck, max_submerged_kt=limit)))
    failures = [g["reason"] for g in gates if not g["passed"] and not g["overridden"]]
    return {"valid": not failures, "failures": failures,
            "overrides": [g["gate"] for g in gates if g["overridden"]],
            "gates": gates, "policy": policy}


@dataclass(frozen=True)
class SubspaceCorrection:
    """Correction factor ``C_S`` at one temperature with its provenance.

    ``log_standard_error`` is the standard error of ``ln C_S``: the RPMD
    statistical and TI-quadrature error plus, for the iterative quantum
    route, the spread of the trapped-norm extrapolation.  Grid, absorbing
    potential and bead-number convergence are not included.

    ``valid`` is false when a validity gate rejected the model
    (:func:`validity_gates`) or the rates do not form a positive ratio.  The
    component rates and the gate record are kept, but ``factor`` is then only
    a diagnostic number: it is written as ``diagnostic_factor`` with
    ``factor: null`` and cannot be applied to a rate.
    """

    beta: float
    dimension: int
    factor: float
    log_standard_error: float
    quantum_rate: float
    rpmd_rate: float
    nbeads: int
    model: Mapping[str, Any]
    quantum: QuantumRateResult | None = None
    rpmd: RingPolymerRateResult | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    valid: bool = True
    validity: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "valid", bool(self.valid))
        usable = np.isfinite(self.factor) and self.factor > 0
        if self.valid and not usable:
            raise ValueError("correction factor must be positive and finite")
        if not usable:
            object.__setattr__(self, "factor", float("nan"))
        object.__setattr__(self, "validity", dict(self.validity))
        if not 1 <= int(self.dimension) <= 3:
            raise ValueError("subspace dimension must be one to three")
        object.__setattr__(self, "model", dict(self.model))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def log_factor(self) -> float:
        return float(np.log(self.factor)) if np.isfinite(self.factor) else float("nan")

    @property
    def invalid_reasons(self) -> list:
        return list(self.validity.get("failures", [])) if not self.valid else []

    def require_valid(self) -> None:
        if not self.valid:
            raise ValueError("the subspace correction failed its validity gates and cannot be applied: "
                             + "; ".join(self.invalid_reasons or ["no reason recorded"]))

    def to_dict(self) -> dict:
        anchored = isinstance(self.diagnostics.get("instanton_anchor"), Mapping)
        approximation = (
            "instanton-anchored subspace defect: k_RPMD(full) * [k_QM(S) / k_RPMD(S)] * [eps_SD(full) / eps_SD(S^0)]"
            if anchored
            else "multiplicative subspace defect: k_RPMD(full) * k_QM(S) / k_RPMD(S)"
        )
        record = {
            "schema": SCHEMA,
            "schema_version": 1,
            "beta": self.beta,
            "dimension": int(self.dimension),
            "valid": self.valid,
            "validity": self.validity,
            "factor": self.factor if self.valid else None,
            "log_factor": self.log_factor if self.valid else None,
            "diagnostic_factor": self.factor,
            "log_standard_error": self.log_standard_error,
            "quantum_rate": self.quantum_rate,
            "rpmd_rate": self.rpmd_rate,
            "nbeads": int(self.nbeads),
            "model": self.model,
            "diagnostics": self.diagnostics,
            "approximation": approximation,
            "exact_general_rate_claim": False,
        }
        if self.quantum is not None:
            record["quantum"] = self.quantum.to_dict()
        if self.rpmd is not None:
            record["rpmd"] = self.rpmd.to_dict()
        return _json_safe(record)

    def write(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SubspaceCorrection":
        if data.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} record")
        def number(value):  # non-finite values of an invalid record are stored as null
            return float("nan") if value is None else float(value)

        factor = data.get("factor")
        if factor is None:
            factor = data.get("diagnostic_factor")
        return cls(float(data["beta"]), int(data["dimension"]), number(factor),
                   number(data["log_standard_error"]), number(data["quantum_rate"]),
                   number(data["rpmd_rate"]), int(data["nbeads"]), data.get("model", {}),
                   diagnostics=data.get("diagnostics", {}),
                   valid=bool(data.get("valid", True)), validity=data.get("validity", {}))

    @classmethod
    def read(cls, path) -> "SubspaceCorrection":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def subspace_correction(
    model: SubspaceModel,
    beta: float,
    nbeads: int,
    *,
    quantum_options: Mapping[str, Any] | None = None,
    rpmd_options: Mapping[str, Any] | None = None,
    rpmd_model: SubspaceModel | None = None,
    execution: Mapping[str, Any] | None = None,
) -> SubspaceCorrection:
    """Compute ``C_S = k_QM(H_S) / k_RPMD(H_S)`` for one temperature.

    ``nbeads`` should match (or exceed) the bead number of the full RPMD
    calculation the factor will multiply, so that ``C_S`` corrects the same
    finite-``P`` estimator.  ``rpmd_model`` (optional) is the same ``H_S`` on
    a finer grid for the ring-polymer spline; the quantum DVR uses ``model``,
    whose nodes must be a subset of it.
    """
    if execution is not None:
        from .subspace_execution import subspace_corrections

        return subspace_corrections({"correction": {
            "model": model, "beta": beta, "nbeads": nbeads, "quantum_options": quantum_options,
            "rpmd_options": rpmd_options, "rpmd_model": rpmd_model,
        }}, execution=execution)["correction"]
    _validate_rpmd_model(model, rpmd_model)
    quantum = exact_quantum_rate(model, beta, **dict(quantum_options or {}))
    rpmd = ring_polymer_rate(rpmd_model if rpmd_model is not None else model, beta, nbeads,
                             **dict(rpmd_options or {}))
    return _assemble_correction(model, beta, nbeads, quantum, rpmd)


def _validate_rpmd_model(model, rpmd_model):
    if rpmd_model is not None:
        if (rpmd_model.dimension != model.dimension or rpmd_model.dividing_surface != model.dividing_surface
                or rpmd_model.reactant_reference != model.reactant_reference):
            raise ValueError("rpmd_model must describe the same subspace, surface and reference")


def _assemble_correction(model, beta, nbeads, quantum, rpmd):
    described = model.describe()
    validity = validity_gates(described["construction"], rpmd.diagnostics)
    positive = bool(np.isfinite(quantum.rate) and np.isfinite(rpmd.rate) and quantum.rate > 0 and rpmd.rate > 0)
    if not positive:
        # Keep both component results: a ratio that cannot be formed is a
        # finding about the model, not a reason to lose the evidence.
        validity["failures"].append("subspace rates must be positive to form a correction factor "
                                    f"(quantum {quantum.rate:.6g}, RPMD {rpmd.rate:.6g})")
        validity["valid"] = False
    quantum_spread = float(quantum.diagnostics.get("relative_extrapolation_spread", 0.0) or 0.0)
    log_error = float(np.hypot(rpmd.log_rate_standard_error, quantum_spread))
    diagnostics = {
        "rpmd_log_standard_error": rpmd.log_rate_standard_error,
        "quantum_relative_extrapolation_spread": quantum_spread,
        "quantum_method": quantum.method,
        "rpmd_transmission": rpmd.transmission,
        "rpmd_qtst_rate": rpmd.qtst_rate,
        "quantum_over_rpmd_qtst": quantum.rate / rpmd.qtst_rate if positive and rpmd.qtst_rate > 0 else None,
        "rpmd_spline_outside_fraction": rpmd.diagnostics.get("spline_outside_fraction"),
        "rpmd_grid_exit": rpmd.diagnostics.get("grid_exit"),
        "units": quantum.diagnostics.get("units"),
    }
    return SubspaceCorrection(float(beta), model.dimension,
                              quantum.rate / rpmd.rate if positive else float("nan"), log_error,
                              quantum.rate, rpmd.rate, int(nbeads), described,
                              quantum, rpmd, diagnostics, validity["valid"], validity)


def harmonic_bead_factor(dividing_frequencies: Any, reactant_frequencies: Any, beta: float, nbeads: int) -> dict:
    """Finite-bead factor ``B_P = k_RPMD(P) / k_RPMD(P -> inf)`` of harmonic environment modes.

    A P-bead ring polymer of a harmonic mode has the partition function
    ``Z_P(u) = 1 / (2 sinh(P asinh(u / 2P)))`` with ``u = beta * omega``
    (exact; ``P = 1`` is classical, ``P -> inf`` quantum).  For the real
    vibrational frequencies at the dividing surface (reaction mode excluded)
    and at the reactants (translations and rotations excluded), in atomic
    units,

        B_P = prod_dividing [Z_P / Z_inf] / prod_reactant [Z_P / Z_inf].

    A subspace correction from an adiabatic projection puts the environment
    zero-point energy into ``H_S`` exactly, so ``C_S`` at matched P does not
    contain the environment's finite-P error; divide the full RPMD rate by
    ``B_P`` before multiplying by ``C_S`` (``apply_subspace_correction(...,
    bead_factor=B_P)``), or converge P.  See docs/subspace-transfer-theory.md.
    """
    beta = float(beta)
    nbeads = int(nbeads)
    if not np.isfinite(beta) or beta <= 0 or nbeads < 1:
        raise ValueError("beta must be positive and nbeads at least one")

    def log_ratio(frequencies):
        omega = np.atleast_1d(np.asarray(frequencies, dtype=float))
        if omega.size and (not np.isfinite(omega).all() or np.any(omega <= 0)):
            raise ValueError("frequencies must be finite and positive (drop imaginary and zero modes)")
        u = beta * omega
        # ln(Z_P / Z_inf) = ln sinh(u/2) - ln sinh(P asinh(u / 2P)), via ln sinh(x) = x + ln(1 - e^{-2x}) - ln 2.
        x_inf, x_p = 0.5 * u, nbeads * np.arcsinh(u / (2 * nbeads))
        values = (x_inf + np.log(-np.expm1(-2 * x_inf))) - (x_p + np.log(-np.expm1(-2 * x_p)))
        return values

    dividing = log_ratio(dividing_frequencies)
    reactant = log_ratio(reactant_frequencies)
    log_factor = float(np.sum(dividing) - np.sum(reactant))
    return {
        "factor": float(np.exp(log_factor)),
        "log_factor": log_factor,
        "nbeads": nbeads,
        "beta": beta,
        "dividing_log_ratios": dividing.tolist(),
        "reactant_log_ratios": reactant.tolist(),
        "approximation": "harmonic environment modes, RPMD-TST level",
    }


def apply_subspace_correction(
    rate: float,
    correction: SubspaceCorrection,
    *,
    standard_error: float | None = None,
    beta: float | None = None,
    beta_rtol: float = 1e-6,
    bead_factor: float | None = None,
    anchor_log_shift: float | None = None,
) -> dict:
    """Multiply a full-dimensional RPMD rate by ``C_S``.

    Relative errors are combined in quadrature assuming the full RPMD run and
    the subspace calculation are statistically independent.  ``beta`` (when
    given) must match the correction's temperature.  ``bead_factor`` (from
    :func:`harmonic_bead_factor`) divides the RPMD rate first, removing the
    environment's finite-bead error that an adiabatic ``H_S`` does not carry.
    ``anchor_log_shift`` (``A_H - A_{S^0}`` from :func:`instanton_anchor`)
    multiplies ``C_S`` by ``exp(anchor_log_shift)`` when supplied explicitly.
    """
    if not isinstance(correction, SubspaceCorrection):
        raise TypeError("correction must be a SubspaceCorrection")
    correction.require_valid()
    relative = None
    rate = float(rate)
    if not np.isfinite(rate) or rate < 0:
        raise ValueError("rate must be finite and nonnegative")
    if beta is not None and not np.isclose(float(beta), correction.beta, rtol=beta_rtol, atol=0):
        raise ValueError(f"correction beta {correction.beta:.12g} does not match the rate's beta {float(beta):.12g}")
    bead = 1.0
    if bead_factor is not None:
        bead = float(bead_factor)
        if not np.isfinite(bead) or bead <= 0:
            raise ValueError("bead_factor must be finite and positive")
    anchor_factor = 1.0
    if anchor_log_shift is not None:
        anchor_log_shift = float(anchor_log_shift)
        if not np.isfinite(anchor_log_shift):
            raise ValueError("anchor_log_shift must be finite")
        anchor_factor = float(np.exp(anchor_log_shift))
    effective_factor = correction.factor * anchor_factor
    corrected = rate * effective_factor / bead
    if standard_error is not None:
        standard_error = float(standard_error)
        if not np.isfinite(standard_error) or standard_error < 0:
            raise ValueError("standard_error must be finite and nonnegative")
        relative = float(np.hypot(standard_error / rate, correction.log_standard_error)) if rate > 0 else None
    result = {
        "rpmd_rate": rate,
        "rpmd_rate_standard_error": standard_error,
        "correction_factor": effective_factor,
        "correction_log_standard_error": correction.log_standard_error,
        "bead_factor": None if bead_factor is None else bead,
        "corrected_rate": corrected,
        "corrected_rate_standard_error": None if relative is None else corrected * relative,
        "corrected_relative_standard_error": relative,
        "subspace_dimension": correction.dimension,
        "beta": correction.beta,
        "approximation": (
            "instanton-anchored multiplicative subspace defect"
            if anchor_log_shift is not None or isinstance(correction.diagnostics.get("instanton_anchor"), Mapping)
            else "multiplicative subspace defect"
        ),
    }
    if anchor_log_shift is not None:
        result["unanchored_correction_factor"] = correction.factor
        result["anchor_log_shift"] = anchor_log_shift
        result["anchor_factor"] = anchor_factor
    return result


def nested_subspace_sensitivity(corrections: Sequence[SubspaceCorrection], *, tolerance: float = 0.05) -> dict:
    """Target-response sensitivity ``ln(C_{d+1} / C_d)`` for nested subspaces.

    Adding a coordinate that RPMD already treats well leaves ``C`` unchanged;
    a coordinate that carries part of the defect changes it.  Each step is

    ``settled``       ``|change| <= tolerance`` and its standard error ``<= tolerance``;
    ``changed``       ``|change| > max(tolerance, 2 sigma)``;
    ``undetermined``  otherwise (too noisy to tell).

    The recommended dimension is the smallest ``d`` whose next step is
    settled, provided no earlier step changed or was undetermined; otherwise
    the largest dimension computed (``recommended_dimension`` is then
    ``None`` when the deciding step was undetermined, so a caller can fall
    back to another criterion).  The caller is responsible for the subspaces
    being nested and at the same temperature.  Invalid corrections are
    refused: a diagnostic factor is no evidence about the defect.
    """
    items = sorted(corrections, key=lambda c: c.dimension)
    for item in items:
        item.require_valid()
    if len(items) < 2:
        raise ValueError("at least two nested corrections are required")
    betas = np.array([c.beta for c in items])
    if not np.allclose(betas, betas[0], rtol=1e-9, atol=0):
        raise ValueError("nested corrections must share one temperature")
    if len({c.dimension for c in items}) != len(items):
        raise ValueError("nested corrections must have distinct dimensions")
    steps = []
    for low, high in zip(items, items[1:]):
        change = high.log_factor - low.log_factor
        sigma = float(np.hypot(low.log_standard_error, high.log_standard_error))
        if abs(change) <= tolerance and sigma <= tolerance:
            status = "settled"
        elif abs(change) > max(tolerance, 2.0 * sigma):
            status = "changed"
        else:
            status = "undetermined"
        steps.append({"from": low.dimension, "to": high.dimension, "log_change": change,
                      "standard_error": sigma, "status": status, "settled": status == "settled"})
    recommended = items[-1].dimension
    for step in steps:
        if step["status"] == "settled":
            recommended = step["from"]
            break
        if step["status"] == "undetermined":
            recommended = None
            break
    return {
        "beta": float(betas[0]),
        "steps": steps,
        "factors": {str(c.dimension): c.factor for c in items},
        "recommended_dimension": recommended,
        "tolerance": float(tolerance),
        "full_budget_used": bool(recommended == items[-1].dimension and steps[-1]["status"] != "settled"),
    }


def _rate_summary(root: Path):
    from .rate_io import _metadata, _read_json

    summary, config, manifest, files = _metadata(root)
    published = root / "results" / "summary.json"
    if published.is_file():
        data = _read_json(published)
        if "rate_extension" in data:
            summary = data
            files.append(published)
    return summary, config, manifest, files


def correct_rate_run(directory, correction, *, output=None, beta_rtol: float = 1e-6) -> dict:
    """Write ``rate_quantum_corrected.json`` next to a saved PyRPMD rate run.

    ``correction`` is a :class:`SubspaceCorrection` or the path of a JSON
    record written by :meth:`SubspaceCorrection.write`.  The run's
    temperature (``input/resolved.json``) must match the correction.  Source
    scientific files are never modified; the output records their SHA-256.
    A run without a publishable ``rate_*`` keeps ``corrected_rate`` empty and
    only reports the corrected point estimate.
    """
    from ..io import file_sha256, write_json
    from .rate_io import _root

    if not isinstance(correction, SubspaceCorrection):
        correction = SubspaceCorrection.read(correction)
    root = _root(directory, "summary.json")
    summary, config, manifest, files = _rate_summary(root)
    task = summary.get("task")
    if task not in ("rate", "surface_rate"):
        raise ValueError("correct_rate_run accepts a gas or surface rate run")
    unit = "s_inv" if task == "surface_rate" else "cm3_s"
    beta = config.get("beta")
    if beta is None:
        raise ValueError("input/resolved.json with beta is required to check the correction temperature")
    if not np.isclose(float(beta), correction.beta, rtol=beta_rtol, atol=0):
        raise ValueError(f"correction beta {correction.beta:.12g} does not match run beta {float(beta):.12g}")
    estimate = summary.get("rate_estimate_" + unit)
    publishable = summary.get("rate_" + unit)
    error = summary.get("rate_standard_error_" + unit)
    if estimate is None:
        raise ValueError("the saved run has no rate estimate to correct")
    applied = apply_subspace_correction(estimate, correction, standard_error=error, beta=beta, beta_rtol=beta_rtol)
    target = Path(output).resolve() if output else root / "rate_quantum_corrected.json"
    protected = {Path(p).resolve() for p in files}
    protected.update((root / name).resolve() for name in (
        "summary.json", "rate_pmf.json", "rate_pmf.npz", "recrossing.npz", "manifest.json",
        "results/summary.json", "input/resolved.json", "input/system.json", "rate_postprocess.json",
    ))
    if target in protected:
        raise ValueError("the corrected rate must not overwrite a source scientific file")
    if target.exists():
        try:
            previous = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("refusing to overwrite an unrelated file") from exc
        if not isinstance(previous, dict) or previous.get("schema") != CORRECTED_SCHEMA:
            raise ValueError("refusing to overwrite an unrelated file")
    result = {
        "schema": CORRECTED_SCHEMA,
        "schema_version": 1,
        "task": task,
        "rate_unit": {"cm3_s": "cm^3 molecule^-1 s^-1", "s_inv": "s^-1"}[unit],
        "source_directory": str(root),
        "source_status": summary.get("status"),
        "source_rate_publishable": publishable is not None,
        "rpmd_rate_estimate_" + unit: estimate,
        "rpmd_rate_standard_error_" + unit: error,
        "corrected_rate_estimate_" + unit: applied["corrected_rate"],
        "corrected_rate_" + unit: applied["corrected_rate"] if publishable is not None else None,
        "corrected_rate_standard_error_" + unit: applied["corrected_rate_standard_error"],
        "correction": correction.to_dict(),
        "physics_fingerprint": manifest.get("physics_fingerprint"),
        "source_files": [{"path": str(p), "sha256": file_sha256(p)} for p in dict.fromkeys(files)],
        "uncertainty_scope": "full-run RPMD standard error and subspace ln C_S error combined in quadrature; grid, absorbing-potential and bead convergence excluded",
        "approximation": applied["approximation"],
        "exact_general_rate_claim": False,
    }
    result = _json_safe(result)
    write_json(target, result)
    return result


def _subspace_mass_weighted_derivatives(model: SubspaceModel):
    """Return vectorized ``(grad_q, hess_q)`` on mass-weighted ``q = sqrt(m) s``."""
    d = model.dimension
    sqrt_m = np.sqrt(model.masses)
    inv_sqrt_m = 1.0 / sqrt_m
    lower = np.array([a[0] for a in model.axes], dtype=float)
    upper = np.array([a[-1] for a in model.axes], dtype=float)
    if model.function is None:
        spline = model.interpolant._spline
        if spline is not None:
            def grad_q(q):
                raw_s = np.asarray(q, dtype=float).reshape(-1, d) * inv_sqrt_m[None, :]
                s = np.clip(raw_s, lower, upper)
                clamped = s != raw_s
                out = np.empty_like(s)
                for a in range(d):
                    order = [0] * d
                    order[a] = 1
                    out[:, a] = spline(s, nu=tuple(order))
                out[clamped] = 0.0
                return out * inv_sqrt_m[None, :]

            def hess_q(q):
                raw_s = np.asarray(q, dtype=float).reshape(-1, d) * inv_sqrt_m[None, :]
                s = np.clip(raw_s, lower, upper)
                clamped = s != raw_s
                out = np.zeros((len(s), d, d), dtype=float)
                for a in range(d):
                    for b in range(a, d):
                        order = [0] * d
                        order[a] += 1
                        order[b] += 1
                        val = np.asarray(spline(s, nu=tuple(order)), dtype=float)
                        val[clamped[:, a] | clamped[:, b]] = 0.0
                        out[:, a, b] = out[:, b, a] = val
                return out * (inv_sqrt_m[None, :, None] * inv_sqrt_m[None, None, :])

            return grad_q, hess_q

    def grad_s_eval(s):
        if model.function is None:
            _, g = model.interpolant.energy_and_gradient(s)
            return g
        if model.gradient is not None:
            return np.asarray(model.gradient(s), dtype=float)
        g = np.empty_like(s)
        for a in range(d):
            step = 1e-4 * float(model.spacing[a])
            shift = np.zeros(d)
            shift[a] = step
            g[:, a] = (np.asarray(model.function(s + shift), float)
                       - np.asarray(model.function(s - shift), float)) / (2.0 * step)
        return g

    def grad_q(q):
        s = np.asarray(q, dtype=float).reshape(-1, d) * inv_sqrt_m[None, :]
        return grad_s_eval(s) * inv_sqrt_m[None, :]

    def hess_q(q):
        s = np.asarray(q, dtype=float).reshape(-1, d) * inv_sqrt_m[None, :]
        h = np.empty((len(s), d, d), dtype=float)
        for a in range(d):
            step = 1e-4 * float(model.spacing[a])
            shift = np.zeros(d)
            shift[a] = step
            h[:, :, a] = (grad_s_eval(s + shift) - grad_s_eval(s - shift)) / (2.0 * step)
        h = 0.5 * (h + h.transpose(0, 2, 1))
        return h * (inv_sqrt_m[None, :, None] * inv_sqrt_m[None, None, :])

    return grad_q, hess_q


def _initial_subspace_saddle_guess(model: SubspaceModel) -> np.ndarray:
    """Locate a barrier minimax grid node near ``model.dividing_surface``.

    In 2D (e.g. ``rph`` or ``plane`` models), tracks the connected transverse
    valley starting near ``(dividing_surface, 0)`` rather than taking a global
    slice minimum across all ``axis1``, so L-shaped pre-reaction or product
    complexes at large transverse displacement do not pull the seed out of the
    barrier channel.
    """
    axis0 = model.axes[0]
    values = model.values
    top_bound = float(values.max())
    valley_j = None
    if model.dimension == 1:
        profile = values
    elif model.dimension == 2:
        axis1 = model.axes[1]
        n0, n1 = values.shape
        i_div = int(np.argmin(np.abs(axis0 - model.dividing_surface)))
        center1 = 0.0 if axis1[0] <= 0.0 <= axis1[-1] else 0.5 * float(axis1[0] + axis1[-1])
        row_div = values[i_div]
        local_mins = [
            j for j in range(1, n1 - 1)
            if row_div[j] <= row_div[j - 1] and row_div[j] <= row_div[j + 1]
            and row_div[j] < top_bound - 1e-9 * max(1.0, abs(top_bound))
        ]
        if local_mins:
            j_div = min(local_mins, key=lambda j: abs(float(axis1[j]) - center1))
        else:
            j_div = int(np.argmin(row_div))
        valley_j = np.empty(n0, dtype=int)
        valley_j[i_div] = j_div
        for i in range(i_div + 1, n0):
            lo, hi = max(0, valley_j[i - 1] - 2), min(n1 - 1, valley_j[i - 1] + 2)
            valley_j[i] = lo + int(np.argmin(values[i, lo : hi + 1]))
        for i in range(i_div - 1, -1, -1):
            lo, hi = max(0, valley_j[i + 1] - 2), min(n1 - 1, valley_j[i + 1] + 2)
            valley_j[i] = lo + int(np.argmin(values[i, lo : hi + 1]))
        profile = values[np.arange(n0), valley_j]
    else:
        flat = values.reshape(len(axis0), -1)
        profile = flat.min(axis=1)
    valid = np.ones(len(axis0), dtype=bool)
    valid[0] = valid[-1] = False
    if model.dimension > 1 and top_bound > float(profile.min()):
        valid &= profile < top_bound - 1e-9 * max(1.0, abs(top_bound))
    indices = np.flatnonzero(valid)
    if not len(indices):
        indices = np.arange(1, len(axis0) - 1)
    peaks = [int(i) for i in indices
             if profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]
             and (profile[i] > profile[i - 1] or profile[i] > profile[i + 1])]
    if peaks:
        prom = profile - np.maximum(
            np.minimum.accumulate(profile),
            np.minimum.accumulate(profile[::-1])[::-1],
        )
        max_prom = max(float(prom[i]) for i in peaks)
        prominent = [i for i in peaks if float(prom[i]) >= max(1e-6, 0.05 * max_prom)] or peaks
        i0 = min(prominent, key=lambda i: abs(float(axis0[i]) - model.dividing_surface))
    else:
        i0 = int(indices[int(np.argmax(profile[indices]))])
    if model.dimension == 1:
        return np.array([float(axis0[i0])], dtype=float)
    if valley_j is not None:
        return np.array([float(axis0[i0]), float(model.axes[1][valley_j[i0]])], dtype=float)
    perp_index = np.unravel_index(int(np.argmin(values[i0])), values.shape[1:])
    return np.array([float(axis0[i0]), *[float(model.axes[k + 1][idx]) for k, idx in enumerate(perp_index)]],
                    dtype=float)


def _p_rfo_saddle_step(g: np.ndarray, vals: np.ndarray, vecs: np.ndarray, step_cap: float) -> np.ndarray:
    """Partitioned rational-function step: uphill along lowest mode, downhill along others."""
    proj = vecs.T @ g
    lam0, g0 = float(vals[0]), float(proj[0])
    mu_max = 0.5 * (lam0 + float(np.hypot(lam0, 2.0 * g0)))
    step = np.zeros_like(proj)
    if abs(lam0 - mu_max) > np.finfo(float).tiny:
        step[0] = -g0 / (lam0 - mu_max)
    if len(vals) > 1:
        m = len(vals)
        augmented = np.zeros((m, m), dtype=float)
        augmented[:-1, :-1] = np.diag(vals[1:])
        augmented[:-1, -1] = augmented[-1, :-1] = proj[1:]
        mu_min = float(np.linalg.eigvalsh(augmented)[0])
        denom = vals[1:] - mu_min
        keep = np.abs(denom) > np.finfo(float).tiny
        step[1:][keep] = -proj[1:][keep] / denom[keep]
    dq = vecs @ step
    size = float(np.linalg.norm(dq))
    if size > step_cap:
        dq *= step_cap / size
    return dq


def _dome_bounded_amplitude(grad_q, q_saddle: np.ndarray, mode: np.ndarray, target_amp: float,
                            *, n_probes: int = 24, min_amp: float = 1e-3) -> float:
    """Cap initial bounce amplitude inside the local barrier dome before any well or wall."""
    if target_amp <= 0.0:
        return target_amp
    alphas = np.linspace(target_amp / n_probes, target_amp, n_probes)
    capped = float(target_amp)
    for sign in (1.0, -1.0):
        pts = q_saddle[None, :] + (sign * alphas)[:, None] * mode[None, :]
        slopes = sign * (grad_q(pts) @ mode)
        # Moving downhill away from a barrier top gives slopes < 0; slopes >= 0 means
        # the ray crossed a pre-reaction / post-reaction minimum or hit a repulsive wall.
        uphill = np.flatnonzero(slopes >= 0.0)
        if len(uphill):
            k = int(uphill[0])
            safe = float(alphas[k - 1]) if k > 0 else 0.5 * float(alphas[0])
            capped = min(capped, 0.75 * safe)
    return max(capped, min(float(target_amp), float(min_amp)))


def _find_subspace_saddle(model: SubspaceModel, grad_q, hess_q, saddle, tolerance: float, max_step: float):
    d = model.dimension
    sqrt_m = np.sqrt(model.masses)
    inv_sqrt_m = 1.0 / sqrt_m
    lower = np.array([a[0] for a in model.axes], dtype=float)
    upper = np.array([a[-1] for a in model.axes], dtype=float)
    if saddle is None:
        s0 = _initial_subspace_saddle_guess(model)
    else:
        s0 = np.asarray(saddle, dtype=float)
        if s0.shape != (d,) or not np.isfinite(s0).all():
            raise ValueError(f"saddle must be a finite sequence of length {d}")
    q = s0 * sqrt_m
    step_cap = max(float(max_step), float(np.linalg.norm(model.spacing * sqrt_m)) * 4.0)
    for _ in range(80):
        g = grad_q(q[None, :])[0]
        h = hess_q(q[None, :])[0]
        if float(np.max(np.abs(g))) <= tolerance:
            break
        vals, vecs = np.linalg.eigh(h)
        dq = _p_rfo_saddle_step(g, vals, vecs, step_cap)
        alpha = 1.0
        in_saddle_basin = vals[0] < 0.0 and (len(vals) == 1 or vals[1] > 0.0)
        if in_saddle_basin and float(np.max(np.abs(g))) > 1e-7:
            norm_g = float(np.linalg.norm(g))
            for _ in range(6):
                trial_g = grad_q((q + alpha * dq)[None, :])[0]
                if float(np.linalg.norm(trial_g)) < norm_g:
                    break
                alpha *= 0.5
        q = q + alpha * dq
        if model.function is None:
            margin = 1e-6 * (upper - lower)
            q = np.clip(q * inv_sqrt_m, lower + margin, upper - margin) * sqrt_m
    g = grad_q(q[None, :])[0]
    h = hess_q(q[None, :])[0]
    vals, vecs = np.linalg.eigh(h)
    s_final = q * inv_sqrt_m
    grad_max = float(np.max(np.abs(g)))
    scale = max(float(np.max(np.abs(vals))), np.finfo(float).tiny)
    neg = int(np.count_nonzero(vals < -1e-7 * scale))
    if model.function is None and np.any((s_final <= lower) | (s_final >= upper)):
        return None, "barrier saddle search hit the subspace grid boundary"
    if grad_max > max(100.0 * tolerance, 1e-6):
        return None, f"barrier saddle search did not converge (|grad|={grad_max:.3g})"
    if neg != 1:
        return None, f"expected a first-order subspace saddle (1 negative mode), found {neg}"
    return {
        "q": q,
        "s": s_final,
        "barrier_frequency": float(np.sqrt(-vals[0])),
        "unstable_mode": vecs[:, 0],
        "hessian_eigenvalues": vals,
    }, None


def subspace_instanton_defect(
    model: SubspaceModel,
    beta: float,
    *,
    use_skeleton: bool = False,
    nbeads: int = 128,
    saddle: Sequence[float] | None = None,
    max_iterations: int = 400,
    tolerance: float = 1e-10,
    max_step: float = 0.3,
    phase_tolerance: float = 0.05,
    raise_on_invalid: bool = False,
) -> dict:
    """Steepest-descent RPMD defect ``eps_SD(H_S)`` (or ``eps_SD(S^0)``) of a :class:`SubspaceModel`.

    Coordinates are transformed internally to mass-weighted ``q = sqrt(m) s``
    (unit mass, ``hbar = 1``).  With ``use_skeleton=True`` the model's bare
    skeleton :meth:`SubspaceModel.skeleton` is used.  Above or at the saddle
    crossover ``beta * omega_b <= 2 * pi`` the ring polymer collapses and
    ``factor = 1, log_factor = 0``.  Below crossover the link-reflected bounce
    orbit is solved by damped Newton and checked on the full ``P * d`` ring
    Hessian with the geometric cyclic-permutation tangent projected out
    (:func:`pyrpmd.instanton._phase_projected_spectrum`), so multi-mode
    breakdowns (such as two-negative-mode SCT instantons at low temperature)
    are flagged with ``valid = False``.
    """
    from ..instanton import InstantonError, _phase_projected_spectrum

    if not isinstance(model, SubspaceModel):
        raise TypeError("model must be a SubspaceModel")
    target = model.skeleton() if use_skeleton else model
    if target.metric is not None:
        raise ValueError("subspace_instanton_defect requires an Euclidean subspace metric (metric=None)")
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if isinstance(nbeads, (bool, np.bool_)) or not isinstance(nbeads, (int, np.integer)) or nbeads < 8 or nbeads % 2:
        raise ValueError("nbeads must be an even integer >= 8")
    nbeads = int(nbeads)
    max_iterations = int(max_iterations)
    tolerance = float(tolerance)
    max_step = float(max_step)
    phase_tolerance = float(phase_tolerance)
    if (max_iterations < 1 or not np.isfinite(tolerance) or tolerance <= 0
            or not np.isfinite(max_step) or max_step <= 0
            or not np.isfinite(phase_tolerance) or phase_tolerance <= 0):
        raise ValueError("max_iterations >= 1 and positive tolerance, max_step, phase_tolerance are required")

    grad_q, hess_q = _subspace_mass_weighted_derivatives(target)
    saddle_info, saddle_err = _find_subspace_saddle(target, grad_q, hess_q, saddle, tolerance, max_step)
    if saddle_info is None:
        if raise_on_invalid:
            raise InstantonError(saddle_err)
        return {
            "regime": "invalid_saddle",
            "below_crossover": None,
            "beta": beta,
            "crossover_beta": None,
            "barrier_frequency": None,
            "saddle": None,
            "factor": float("nan"),
            "log_factor": float("nan"),
            "eta1": None,
            "nbeads": nbeads,
            "negative_modes": None,
            "converged": False,
            "used_skeleton": bool(use_skeleton),
            "valid": False,
            "failure_reason": saddle_err,
        }

    omega_b = saddle_info["barrier_frequency"]
    crossover_beta = 2.0 * np.pi / omega_b
    q_saddle = saddle_info["q"]
    s_saddle = saddle_info["s"]
    mode = saddle_info["unstable_mode"]

    if beta * omega_b <= 2.0 * np.pi * (1.0 + 1e-6):
        return {
            "regime": "above_crossover",
            "below_crossover": False,
            "beta": beta,
            "crossover_beta": float(crossover_beta),
            "barrier_frequency": float(omega_b),
            "saddle": s_saddle.tolist(),
            "factor": 1.0,
            "log_factor": 0.0,
            "eta1": None,
            "nbeads": nbeads,
            "negative_modes": 0,
            "converged": True,
            "used_skeleton": bool(use_skeleton),
            "valid": True,
            "failure_reason": None,
        }

    d = target.dimension
    half = nbeads // 2
    mapping = np.concatenate((np.arange(half), np.arange(half - 1, -1, -1)))
    omega_n = nbeads / beta
    osc_length = 1.0 / np.sqrt(max(omega_b, 1e-12))
    sqrt_m = np.sqrt(target.masses)
    inv_sqrt_m = 1.0 / sqrt_m
    lower = np.array([a[0] for a in target.axes], dtype=float)
    upper = np.array([a[-1] for a in target.axes], dtype=float)
    grid_span = float(np.min((upper - lower) * sqrt_m))
    base_amp = min(2.4 * osc_length, 0.25 * grid_span)
    amplitude = base_amp * np.sqrt(max(1.0 - (2.0 * np.pi / (beta * omega_b)) ** 2, 0.02))
    amplitude = _dome_bounded_amplitude(grad_q, q_saddle, mode, amplitude)
    angles = 2.0 * np.pi * (np.arange(half) + 0.5) / nbeads
    x_half = q_saddle[None, :] + amplitude * np.cos(angles)[:, None] * mode[None, :]
    step_limit = max(max_step, 0.5 * osc_length)

    def half_grad_and_jac(x):
        g_loc = grad_q(x)
        h_loc = hess_q(x)
        spring = np.empty_like(x)
        if half == 1:  # pragma: no cover - nbeads >= 8 guarantees half >= 4
            spring[0] = 0.0
        else:
            spring[0] = x[0] - x[1]
            spring[-1] = x[-1] - x[-2]
            spring[1:-1] = 2.0 * x[1:-1] - x[:-2] - x[2:]
        g_total = g_loc + omega_n**2 * spring
        jac = np.zeros((half, d, half, d), dtype=float)
        eye = omega_n**2 * np.eye(d)
        for k in range(half):
            jac[k, :, k, :] = h_loc[k] + (eye if k in (0, half - 1) else 2.0 * eye)
            if k > 0:
                jac[k, :, k - 1, :] -= eye
            if k < half - 1:
                jac[k, :, k + 1, :] -= eye
        return g_total, jac.reshape(half * d, half * d)

    converged = False
    g_half = None
    for _ in range(max_iterations):
        g_half, jac = half_grad_and_jac(x_half)
        grad_max = float(np.max(np.abs(g_half)))
        if grad_max <= tolerance:
            converged = True
            break
        vals, vecs = np.linalg.eigh(jac)
        step = _p_rfo_saddle_step(g_half.ravel(), vals, vecs, step_limit * np.sqrt(half)).reshape(x_half.shape)
        size = float(np.max(np.abs(step)))
        if size > step_limit:
            step *= step_limit / size
        scale_h = max(abs(float(vals[0])), np.finfo(float).tiny)
        neg_half = int(np.count_nonzero(vals < -1e-5 * scale_h))
        alpha = 1.0
        if neg_half == 1 and grad_max > 1e-7:
            g_norm = float(np.linalg.norm(g_half))
            for _ in range(5):
                trial_g, _ = half_grad_and_jac(x_half + alpha * step)
                if float(np.linalg.norm(trial_g)) < g_norm or float(np.max(np.abs(trial_g))) <= tolerance:
                    break
                alpha *= 0.5
        x_half = x_half + alpha * step
        if float(np.max(np.abs(alpha * step))) <= 1e-12 * max(1.0, float(np.max(np.abs(x_half)))):
            g_half, _ = half_grad_and_jac(x_half)
            converged = float(np.max(np.abs(g_half))) <= max(tolerance, 1e-8)
            break

    if g_half is None:
        g_half, _ = half_grad_and_jac(x_half)
    q_ring = x_half[mapping]
    h_ring_loc = hess_q(x_half)[mapping]
    h_ring = np.zeros((nbeads, d, nbeads, d), dtype=float)
    eye = omega_n**2 * np.eye(d)
    for j in range(nbeads):
        h_ring[j, :, j, :] += h_ring_loc[j] + 2.0 * eye
        h_ring[j, :, (j - 1) % nbeads, :] -= eye
        h_ring[j, :, (j + 1) % nbeads, :] -= eye
    h_ring = h_ring.reshape(nbeads * d, nbeads * d)

    spread = float(np.max(np.ptp(q_ring, axis=0)))
    collapsed = spread <= 1e-4 * min(1.0, osc_length)
    s_ring = q_ring * inv_sqrt_m[None, :]
    exited_grid = bool(target.function is None and np.any((s_ring <= lower) | (s_ring >= upper)))

    raw_spectrum = np.linalg.eigvalsh(h_ring)
    curvature_scale = max(abs(float(raw_spectrum[0])), np.finfo(float).tiny)
    mode_cutoff = 1e-4 * curvature_scale
    raw_neg_modes = int(np.count_nonzero(raw_spectrum < -mode_cutoff))

    freqs = np.fft.fftfreq(nbeads)
    tangent = np.fft.ifft(2j * np.pi * freqs[:, None] * np.fft.fft(q_ring, axis=0), axis=0).real.ravel()
    tangent_norm = float(np.linalg.norm(tangent))
    phase_residual = (float(np.linalg.norm(h_ring @ tangent))
                      / max(curvature_scale * tangent_norm, np.finfo(float).tiny))
    if not collapsed and tangent_norm > 1e-12 and (raw_neg_modes <= 1 or phase_residual <= phase_tolerance):
        spectrum = _phase_projected_spectrum(h_ring, tangent)
    else:
        spectrum = raw_spectrum
    neg_modes = int(np.count_nonzero(spectrum < -mode_cutoff))
    eta1 = float(np.sqrt(-spectrum[0])) if spectrum[0] < 0.0 else None

    reasons = []
    if not converged:
        reasons.append(f"subspace instanton did not converge (|grad|={float(np.max(np.abs(g_half))):.3g})")
    if collapsed:
        reasons.append("subspace instanton collapsed onto the barrier saddle")
    if exited_grid:
        reasons.append("subspace instanton reached the grid boundary")
    if raw_neg_modes > 1 and phase_residual > phase_tolerance:
        reasons.append(
            f"imaginary-time phase residual {phase_residual:.3g} exceeds phase_tolerance {phase_tolerance:.3g}; "
            "increase nbeads to resolve the phase mode"
        )
    if neg_modes != 1:
        reasons.append(f"expected one physical negative instanton mode, found {neg_modes}")
    if eta1 is None or not np.isfinite(eta1) or eta1 <= 0.0:
        reasons.append("subspace instanton unstable frequency eta1 is not positive")

    if reasons:
        reason = "; ".join(reasons)
        if raise_on_invalid:
            raise InstantonError(reason)
        return {
            "regime": "below_crossover",
            "below_crossover": True,
            "beta": beta,
            "crossover_beta": float(crossover_beta),
            "barrier_frequency": float(omega_b),
            "saddle": s_saddle.tolist(),
            "factor": float("nan"),
            "log_factor": float("nan"),
            "eta1": eta1,
            "nbeads": nbeads,
            "spread": spread,
            "gradient_max": float(np.max(np.abs(g_half))),
            "negative_modes": neg_modes,
            "raw_negative_modes": raw_neg_modes,
            "phase_mode_relative_residual": phase_residual,
            "exited_grid": exited_grid,
            "converged": converged,
            "used_skeleton": bool(use_skeleton),
            "valid": False,
            "failure_reason": reason,
        }

    factor = float(2.0 * np.pi / (beta * eta1))
    return {
        "regime": "below_crossover",
        "below_crossover": True,
        "beta": beta,
        "crossover_beta": float(crossover_beta),
        "barrier_frequency": float(omega_b),
        "saddle": s_saddle.tolist(),
        "factor": factor,
        "log_factor": float(np.log(factor)),
        "eta1": eta1,
        "nbeads": nbeads,
        "spread": spread,
        "gradient_max": float(np.max(np.abs(g_half))),
        "negative_modes": neg_modes,
        "raw_negative_modes": raw_neg_modes,
        "phase_mode_relative_residual": phase_residual,
        "exited_grid": exited_grid,
        "converged": True,
        "used_skeleton": bool(use_skeleton),
        "valid": True,
        "failure_reason": None,
    }


def _ring_rigid_modes(q_ring: np.ndarray, masses_coord: np.ndarray, rigid_body: bool) -> np.ndarray:
    """Orthonormal global translation and rigid-rotation zero modes of a 3D ring polymer."""
    nbeads, d = q_ring.shape
    if not rigid_body:
        return np.zeros((nbeads * d, 0), dtype=float)
    if d % 3 != 0:
        raise ValueError("rigid_body=True requires 3N Cartesian coordinates")
    n_atoms = d // 3
    m_atom = np.asarray(masses_coord, dtype=float).reshape(n_atoms, 3)[:, 0]
    sqrt_m_atom = np.sqrt(m_atom)
    x_ring = (q_ring / np.sqrt(masses_coord)[None, :]).reshape(nbeads, n_atoms, 3)
    com = np.sum(x_ring * m_atom[None, :, None], axis=(0, 1)) / (nbeads * float(np.sum(m_atom)))
    rel = x_ring - com[None, None, :]
    basis = []
    for axis in range(3):
        trans = np.zeros((nbeads, n_atoms, 3), dtype=float)
        trans[:, :, axis] = sqrt_m_atom[None, :]
        basis.append(trans.ravel())
    for axis in range(3):
        e = np.zeros(3, dtype=float)
        e[axis] = 1.0
        rot = np.cross(e[None, None, :], rel) * sqrt_m_atom[None, :, None]
        basis.append(rot.ravel())
    mat = np.column_stack(basis)
    u, s, _ = np.linalg.svd(mat, full_matrices=False)
    if not len(s) or s[0] <= 1e-12:
        return np.zeros((nbeads * d, 0), dtype=float)
    return u[:, s > 1e-8 * s[0]]


def _projected_ring_spectrum(h_blocks: np.ndarray, omega_n: float, zero_basis: np.ndarray):
    """Lowest physical eigenvalues of a ring Hessian restricted to ``zero_basis`` complement."""
    nbeads, d, _ = h_blocks.shape
    total = nbeads * d
    u = np.asarray(zero_basis, dtype=float).reshape(total, -1)
    if u.shape[1]:
        u, s, _ = np.linalg.svd(u, full_matrices=False)
        u = u[:, s > 1e-8 * max(float(s[0]), 1.0)]
    diag_max = float(np.max(np.abs(np.diagonal(h_blocks, axis1=1, axis2=2)))) + 2.0 * omega_n**2
    shift = max(2.0 * diag_max, 1.0)

    if total <= 2048:
        h_dense = np.zeros((nbeads, d, nbeads, d), dtype=float)
        eye = omega_n**2 * np.eye(d)
        for j in range(nbeads):
            h_dense[j, :, j, :] += h_blocks[j] + 2.0 * eye
            h_dense[j, :, (j - 1) % nbeads, :] -= eye
            h_dense[j, :, (j + 1) % nbeads, :] -= eye
        mat = h_dense.reshape(total, total)
        if u.shape[1]:
            hu = mat @ u
            ut_hu = u.T @ hu
            mat = mat - u @ hu.T - hu @ u.T + u @ ((ut_hu + shift * np.eye(u.shape[1])) @ u.T)
        return np.linalg.eigvalsh(mat), shift

    from scipy.sparse.linalg import LinearOperator, eigsh

    def ring_matvec(vec):
        v = vec.reshape(nbeads, d)
        out = np.einsum("bij,bj->bi", h_blocks, v) + omega_n**2 * (
            2.0 * v - np.roll(v, 1, axis=0) - np.roll(v, -1, axis=0)
        )
        return out.ravel()

    def proj_matvec(vec):
        if not u.shape[1]:
            return ring_matvec(vec)
        coeff = u.T @ vec
        v_perp = vec - u @ coeff
        hv = ring_matvec(v_perp)
        return hv - u @ (u.T @ hv) + shift * (u @ coeff)

    k = min(8, total - 2)
    op = LinearOperator((total, total), matvec=proj_matvec, dtype=float)
    vals = eigsh(op, k=k, which="SA", return_eigenvectors=False, tol=1e-9)
    return np.sort(vals), shift


def full_instanton_defect(
    energy,
    gradient,
    masses: Sequence[float] | np.ndarray,
    beta: float,
    saddle: Sequence[float] | np.ndarray,
    *,
    rigid_body: bool = True,
    refine: bool = True,
    nbeads: int = 64,
    hessian=None,
    gradient_batch=None,
    hessian_displacement: float = 1e-3,
    ts_gradient_tolerance: float = 1e-5,
    tolerance: float = 1e-8,
    max_iterations: int = 300,
    hessian_update_interval: int = 8,
    trust_radius: float = 0.5,
    phase_tolerance: float = 0.05,
    raise_on_invalid: bool = False,
) -> dict:
    """Steepest-descent RPMD defect ``eps_SD(H)`` of a general ``D``-dimensional potential.

    Supports both molecular ``3N`` Cartesian potentials (``rigid_body=True``,
    global translations and rotations removed) and abstract or orthogonally
    rotated ``D``-dimensional model potentials (``rigid_body=False``).  When
    ``refine=True`` the supplied transition-state guess ``saddle`` is first
    refined to a stationary first-order saddle by partitioned RFO, and the
    initial half-ring bounce seed is bounded inside the local barrier dome so
    pre-reaction wells and complexes do not trap the seed.
    """
    from ..constants import KB_AU
    from ..instanton import InstantonError, _bofill_update
    from .subspace_paths import _vibrational_basis, mass_weighted_hessian, refine_saddle

    x0 = np.asarray(saddle, dtype=float).ravel()
    d = x0.size
    if d < 1 or not np.isfinite(x0).all():
        raise ValueError("saddle must be a finite nonempty coordinate array")
    raw_masses = np.asarray(masses, dtype=float).ravel()
    if raw_masses.size == d:
        coord_masses = raw_masses.copy()
    elif rigid_body and 3 * raw_masses.size == d:
        coord_masses = np.repeat(raw_masses, 3)
    else:
        raise ValueError("masses must match saddle length D (or N atoms when D == 3N and rigid_body=True)")
    if not np.isfinite(coord_masses).all() or np.any(coord_masses <= 0):
        raise ValueError("masses must be finite and positive")
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if isinstance(nbeads, (bool, np.bool_)) or not isinstance(nbeads, (int, np.integer)) or nbeads < 8 or nbeads % 2:
        raise ValueError("nbeads must be an even integer >= 8")
    nbeads = int(nbeads)
    temperature_k = 1.0 / (KB_AU * beta)
    sqrt_m = np.sqrt(coord_masses)
    inv_sqrt_m = 1.0 / sqrt_m

    if refine:
        x_ts, ts_rec = refine_saddle(
            gradient, x0, coord_masses, rigid_body=rigid_body,
            gradient_tolerance=float(ts_gradient_tolerance),
            displacement=float(hessian_displacement),
        )
        if not ts_rec["saddle"]:
            reason = (
                f"saddle refinement did not find a first-order saddle "
                f"(negative_modes={ts_rec['negative_modes']}, |grad|={ts_rec['gradient_max']:.3g})"
            )
            if raise_on_invalid:
                raise InstantonError(reason)
            return {
                "regime": "invalid_saddle", "below_crossover": None, "beta": beta,
                "temperature_kelvin": temperature_k, "crossover_beta": None,
                "crossover_temperature_kelvin": None, "barrier_frequency": None,
                "factor": float("nan"), "log_factor": float("nan"), "eta1": None,
                "nbeads": nbeads, "valid": False, "failure_reason": reason,
            }
        omega_b = float(ts_rec["imaginary_frequency"])
        mode = np.asarray(ts_rec["mode"], dtype=float)
        mode /= np.linalg.norm(mode)
    else:
        x_ts = x0.copy()
        basis = _vibrational_basis(x_ts, coord_masses, rigid_body)
        g_vib = basis.T @ (np.asarray(gradient(x_ts), dtype=float).ravel() * inv_sqrt_m)
        grad_cart = float(np.max(np.abs((basis @ g_vib) * sqrt_m))) if g_vib.size else 0.0
        h_mw = (
            np.asarray(hessian(x_ts), dtype=float).reshape(d, d) * (inv_sqrt_m[:, None] * inv_sqrt_m[None, :])
            if hessian is not None
            else mass_weighted_hessian(gradient, x_ts, coord_masses, displacement=float(hessian_displacement))
        )
        vals, vecs = np.linalg.eigh(basis.T @ h_mw @ basis)
        scale = max(float(np.max(np.abs(vals))) if vals.size else 0.0, np.finfo(float).tiny)
        neg = int(np.count_nonzero(vals < -1e-7 * scale))
        if grad_cart > max(10.0 * float(ts_gradient_tolerance), 1e-5) or neg != 1:
            reason = f"supplied saddle is not a stationary first-order saddle (negative_modes={neg}, |grad|={grad_cart:.3g})"
            if raise_on_invalid:
                raise InstantonError(reason)
            return {
                "regime": "invalid_saddle", "below_crossover": None, "beta": beta,
                "temperature_kelvin": temperature_k, "crossover_beta": None,
                "crossover_temperature_kelvin": None, "barrier_frequency": None,
                "factor": float("nan"), "log_factor": float("nan"), "eta1": None,
                "nbeads": nbeads, "valid": False, "failure_reason": reason,
            }
        omega_b = float(np.sqrt(-vals[0]))
        mode = basis @ vecs[:, 0]
        mode /= np.linalg.norm(mode)

    crossover_beta = 2.0 * np.pi / omega_b
    crossover_k = 1.0 / (KB_AU * crossover_beta)
    if beta * omega_b <= 2.0 * np.pi * (1.0 + 1e-6):
        return {
            "regime": "above_crossover",
            "below_crossover": False,
            "beta": beta,
            "temperature_kelvin": temperature_k,
            "crossover_beta": float(crossover_beta),
            "crossover_temperature_kelvin": float(crossover_k),
            "barrier_frequency": float(omega_b),
            "saddle": x_ts.tolist(),
            "factor": 1.0,
            "log_factor": 0.0,
            "eta1": None,
            "nbeads": nbeads,
            "negative_modes": 0,
            "converged": True,
            "valid": True,
            "failure_reason": None,
        }

    def grad_q(q_batch):
        q_arr = np.asarray(q_batch, dtype=float).reshape(-1, d)
        x_arr = q_arr * inv_sqrt_m[None, :]
        if gradient_batch is not None:
            g_arr = np.asarray(gradient_batch(x_arr), dtype=float).reshape(-1, d)
        else:
            g_arr = np.vstack([np.asarray(gradient(row), dtype=float).ravel() for row in x_arr])
        return g_arr * inv_sqrt_m[None, :]

    def hess_q(q_batch):
        q_arr = np.asarray(q_batch, dtype=float).reshape(-1, d)
        n_b = len(q_arr)
        if hessian is not None:
            x_arr = q_arr * inv_sqrt_m[None, :]
            out = np.empty((n_b, d, d), dtype=float)
            for idx, row in enumerate(x_arr):
                h_phys = np.asarray(hessian(row), dtype=float).reshape(d, d)
                out[idx] = 0.5 * (h_phys + h_phys.T) * (inv_sqrt_m[:, None] * inv_sqrt_m[None, :])
            return out
        step_mw = float(hessian_displacement) * float(np.min(sqrt_m))
        eye = step_mw * np.eye(d)
        plus = (q_arr[:, None, :] + eye[None, :, :]).reshape(n_b * d, d)
        minus = (q_arr[:, None, :] - eye[None, :, :]).reshape(n_b * d, d)
        gp = grad_q(plus).reshape(n_b, d, d)
        gm = grad_q(minus).reshape(n_b, d, d)
        h_mw = (gp - gm) / (2.0 * step_mw)
        return 0.5 * (h_mw + h_mw.transpose(0, 2, 1))

    half = nbeads // 2
    mapping = np.concatenate((np.arange(half), np.arange(half - 1, -1, -1)))
    omega_n = nbeads / beta
    q_saddle = x_ts * sqrt_m
    osc_length = 1.0 / np.sqrt(max(omega_b, 1e-12))
    base_amp = 2.2 * osc_length
    amplitude = base_amp * np.sqrt(max(1.0 - (2.0 * np.pi / (beta * omega_b)) ** 2, 0.02))
    amplitude = _dome_bounded_amplitude(grad_q, q_saddle, mode, amplitude)
    angles = 2.0 * np.pi * (np.arange(half) + 0.5) / nbeads
    x_half = q_saddle[None, :] + amplitude * np.cos(angles)[:, None] * mode[None, :]

    def half_grad(x):
        g_loc = grad_q(x)
        spring = np.empty_like(x)
        spring[0] = x[0] - x[1]
        spring[-1] = x[-1] - x[-2]
        spring[1:-1] = 2.0 * x[1:-1] - x[:-2] - x[2:]
        return (g_loc + omega_n**2 * spring).ravel()

    def half_jac(x):
        h_loc = hess_q(x)
        jac = np.zeros((half, d, half, d), dtype=float)
        eye = omega_n**2 * np.eye(d)
        for k in range(half):
            jac[k, :, k, :] = h_loc[k] + (eye if k in (0, half - 1) else 2.0 * eye)
            if k > 0:
                jac[k, :, k - 1, :] -= eye
            if k < half - 1:
                jac[k, :, k + 1, :] -= eye
        return jac.reshape(half * d, half * d)

    radius = max(float(trust_radius) * float(np.mean(sqrt_m)), 0.5 * osc_length)
    max_radius = radius
    z = x_half.ravel()

    def projected_half_grad(z_vec):
        xh = z_vec.reshape(half, d)
        raw_g = half_grad(xh)
        rb = _ring_rigid_modes(xh, coord_masses, rigid_body)
        if rb.shape[1]:
            raw_g = raw_g - rb @ (rb.T @ raw_g)
        return raw_g, rb

    g, rb_half = projected_half_grad(z)
    residual = float(np.max(np.abs(g.reshape(half, d) * sqrt_m[None, :])))
    converged = residual <= tolerance
    if not converged:
        h_mat = half_jac(x_half)
        fresh_hessian = True
        since_refresh = 0
        for _ in range(int(max_iterations)):
            h_work = h_mat
            if rb_half.shape[1]:
                shift = max(2.0 * float(np.max(np.abs(np.diag(h_mat)))), 1.0)
                hu = h_mat @ rb_half
                h_work = (
                    h_mat - rb_half @ hu.T - hu @ rb_half.T
                    + rb_half @ ((rb_half.T @ hu + shift * np.eye(rb_half.shape[1])) @ rb_half.T)
                )
            vals, vecs = np.linalg.eigh(h_work)
            scale_h = max(abs(float(vals[0])), np.finfo(float).tiny)
            neg_half = int(np.count_nonzero(vals < -1e-5 * scale_h))
            delta = _p_rfo_saddle_step(g, vals, vecs, radius * np.sqrt(half))
            if rb_half.shape[1]:
                delta = delta - rb_half @ (rb_half.T @ delta)
            bead_step = float(np.max(np.linalg.norm(delta.reshape(half, d), axis=1)))
            if bead_step <= 1e-13 * max(1.0, float(np.max(np.abs(z)))):
                converged = residual <= max(tolerance, 1e-7)
                break
            if bead_step > radius:
                delta *= radius / bead_step
            if neg_half == 1:
                g_norm = float(np.linalg.norm(g))
                alpha = 1.0
                accepted = False
                for _ in range(5):
                    trial_z = z + alpha * delta
                    trial_g, trial_rb = projected_half_grad(trial_z)
                    trial_res = float(np.max(np.abs(trial_g.reshape(half, d) * sqrt_m[None, :])))
                    if float(np.linalg.norm(trial_g)) < g_norm or trial_res <= tolerance:
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted and not fresh_hessian:
                    h_mat = half_jac(z.reshape(half, d))
                    radius = max(radius, 0.25 * max_radius)
                    fresh_hessian = True
                    since_refresh = 0
                    continue
                step_taken = (alpha * delta) if accepted else (0.25 * delta)
                if not accepted:
                    trial_z = z + step_taken
                    trial_g, trial_rb = projected_half_grad(trial_z)
                    trial_res = float(np.max(np.abs(trial_g.reshape(half, d) * sqrt_m[None, :])))
                    radius = max(0.5 * radius, 1e-6)
                elif alpha == 1.0 and bead_step >= 0.9 * radius:
                    radius = min(max_radius, 1.5 * radius)
                h_mat, _ = _bofill_update(h_mat, step_taken, trial_g - g)
                z, g, rb_half, residual = trial_z, trial_g, trial_rb, trial_res
                fresh_hessian = False
                since_refresh += 1
                if residual <= tolerance:
                    converged = True
                    break
                if hessian_update_interval and since_refresh >= int(hessian_update_interval):
                    h_mat = half_jac(z.reshape(half, d))
                    fresh_hessian = True
                    since_refresh = 0
            else:
                # Outside the 1-negative-mode bounce basin (e.g. expanding from a
                # dome-bounded seed across the action inflection point): take the
                # P-RFO saddle step and refresh the exact half-ring Hessian.
                trial_z = z + delta
                trial_g, trial_rb = projected_half_grad(trial_z)
                trial_res = float(np.max(np.abs(trial_g.reshape(half, d) * sqrt_m[None, :])))
                z, g, rb_half, residual = trial_z, trial_g, trial_rb, trial_res
                if residual <= tolerance:
                    converged = True
                    break
                h_mat = half_jac(z.reshape(half, d))
                fresh_hessian = True
                since_refresh = 0

    x_half = z.reshape(half, d)
    q_ring = x_half[mapping]
    h_blocks = hess_q(x_half)[mapping]
    spread = float(np.max(np.ptp(q_ring, axis=0)))
    collapsed = spread <= 1e-4 * min(1.0, osc_length)

    rigid_basis = _ring_rigid_modes(q_ring, coord_masses, rigid_body)
    raw_spec, _ = _projected_ring_spectrum(h_blocks, omega_n, rigid_basis)
    curvature_scale = max(abs(float(raw_spec[0])), np.finfo(float).tiny)
    mode_cutoff = 1e-4 * curvature_scale
    raw_neg_modes = int(np.count_nonzero(raw_spec < -mode_cutoff))

    freqs = np.fft.fftfreq(nbeads)
    tangent = np.fft.ifft(2j * np.pi * freqs[:, None] * np.fft.fft(q_ring, axis=0), axis=0).real.ravel()
    tangent_norm = float(np.linalg.norm(tangent))
    t_ring = tangent.reshape(nbeads, d)
    ht_ring = np.einsum("bij,bj->bi", h_blocks, t_ring) + omega_n**2 * (
        2.0 * t_ring - np.roll(t_ring, 1, axis=0) - np.roll(t_ring, -1, axis=0)
    )
    phase_residual = float(np.linalg.norm(ht_ring)) / max(curvature_scale * tangent_norm, np.finfo(float).tiny)

    if not collapsed and tangent_norm > 1e-12 and (raw_neg_modes <= 1 or phase_residual <= phase_tolerance):
        zero_basis = np.column_stack([rigid_basis, tangent / tangent_norm]) if rigid_basis.shape[1] else (tangent / tangent_norm)[:, None]
        spectrum, _ = _projected_ring_spectrum(h_blocks, omega_n, zero_basis)
    else:
        spectrum = raw_spec
    neg_modes = int(np.count_nonzero(spectrum < -mode_cutoff))
    eta1 = float(np.sqrt(-spectrum[0])) if spectrum[0] < 0.0 else None

    reasons = []
    if not converged:
        reasons.append(f"full instanton did not converge (|grad|={residual:.3g})")
    if collapsed:
        reasons.append("full instanton collapsed onto the barrier saddle")
    if raw_neg_modes > 1 and phase_residual > phase_tolerance:
        reasons.append(
            f"imaginary-time phase residual {phase_residual:.3g} exceeds phase_tolerance {phase_tolerance:.3g}; "
            "increase nbeads to resolve the phase mode"
        )
    if neg_modes != 1:
        reasons.append(f"expected one physical negative instanton mode, found {neg_modes}")
    if eta1 is None or not np.isfinite(eta1) or eta1 <= 0.0:
        reasons.append("full instanton unstable frequency eta1 is not positive")

    if reasons:
        reason = "; ".join(reasons)
        if raise_on_invalid:
            raise InstantonError(reason)
        return {
            "regime": "below_crossover",
            "below_crossover": True,
            "beta": beta,
            "temperature_kelvin": temperature_k,
            "crossover_beta": float(crossover_beta),
            "crossover_temperature_kelvin": float(crossover_k),
            "barrier_frequency": float(omega_b),
            "factor": float("nan"),
            "log_factor": float("nan"),
            "eta1": eta1,
            "nbeads": nbeads,
            "spread": spread,
            "gradient_max": residual,
            "negative_modes": neg_modes,
            "raw_negative_modes": raw_neg_modes,
            "phase_mode_relative_residual": phase_residual,
            "converged": converged,
            "valid": False,
            "failure_reason": reason,
        }

    factor = float(2.0 * np.pi / (beta * eta1))
    return {
        "regime": "below_crossover",
        "below_crossover": True,
        "beta": beta,
        "temperature_kelvin": temperature_k,
        "crossover_beta": float(crossover_beta),
        "crossover_temperature_kelvin": float(crossover_k),
        "barrier_frequency": float(omega_b),
        "factor": factor,
        "log_factor": float(np.log(factor)),
        "eta1": eta1,
        "nbeads": nbeads,
        "spread": spread,
        "gradient_max": residual,
        "negative_modes": neg_modes,
        "raw_negative_modes": raw_neg_modes,
        "phase_mode_relative_residual": phase_residual,
        "converged": True,
        "valid": True,
        "failure_reason": None,
    }


def _normalize_defect_record(item, beta: float, *, use_skeleton: bool, nbeads: int, beta_rtol: float, label: str) -> dict:
    from ..instanton import InstantonResult, steepest_descent_defect

    if isinstance(item, SubspaceModel):
        record = subspace_instanton_defect(item, beta, use_skeleton=use_skeleton, nbeads=nbeads)
        if (
            not record.get("valid")
            and record.get("converged")
            and item.function is None
            and nbeads < 384
            and (record.get("raw_negative_modes") or 0) > 1
            and (record.get("phase_mode_relative_residual") or 0.0) > 0.05
        ):
            record = subspace_instanton_defect(item, beta, use_skeleton=use_skeleton, nbeads=384)
    elif isinstance(item, InstantonResult):
        record = steepest_descent_defect(item, beta=beta, raise_on_invalid=False)
    elif isinstance(item, Mapping):
        record = dict(item)
    else:
        raise TypeError(f"{label} must be a defect mapping, InstantonResult or SubspaceModel")
    if "beta" in record and record["beta"] is not None:
        if not np.isclose(float(record["beta"]), beta, rtol=beta_rtol, atol=0.0):
            raise ValueError(f"{label} beta {float(record['beta']):.12g} does not match correction beta {beta:.12g}")
    return record


def instanton_anchor(
    correction: SubspaceCorrection,
    full_defect: Mapping[str, Any] | Any,
    skeleton_defect: Mapping[str, Any] | SubspaceModel | None = None,
    *,
    naive_defect: Mapping[str, Any] | SubspaceModel | None = None,
    nbeads: int = 128,
    beta_rtol: float = 1e-6,
    crossover_rtol: float = 0.05,
    allow_crossover_mismatch: bool = False,
) -> dict:
    """Anchor a :class:`SubspaceCorrection` to the full-system steepest-descent defect.

    Implements the skeleton-consistent anchor of ``docs/subspace-instanton-anchor.md``:

        ln C_anch^skel = ln C_S + A_H - A_{S^0},      A_X = ln eps_SD(X),

    where ``S^0`` is the bare-potential skeleton of ``H_S`` (same path and
    coordinate mass without transverse zero-point / free-energy dressing).
    Comparing ``A_H`` against ``A_{S^0}`` preserves the first-order coupling
    theorem for dressed projections while restoring the missing steepest-descent
    defect from path curvature, slow bath coupling, or coordinate-mass shifts.

    When the temperature lies between the crossover temperatures of ``H`` and
    ``S^0`` (one below crossover, one above), the kinks of ``A_H`` and ``A_{S^0}``
    are misaligned and the anchor is marked invalid unless
    ``allow_crossover_mismatch=True``.
    """
    if not isinstance(correction, SubspaceCorrection):
        raise TypeError("correction must be a SubspaceCorrection")
    correction.require_valid()
    if skeleton_defect is None:
        raise ValueError("skeleton_defect (defect mapping or SubspaceModel) is required")
    beta = float(correction.beta)
    full_rec = _normalize_defect_record(full_defect, beta, use_skeleton=False, nbeads=nbeads,
                                        beta_rtol=beta_rtol, label="full_defect")
    skel_rec = _normalize_defect_record(skeleton_defect, beta, use_skeleton=True, nbeads=nbeads,
                                        beta_rtol=beta_rtol, label="skeleton_defect")
    naive_rec = None
    if naive_defect is not None:
        naive_rec = _normalize_defect_record(naive_defect, beta, use_skeleton=False, nbeads=nbeads,
                                             beta_rtol=beta_rtol, label="naive_defect")

    failures = []
    warnings = []
    if not full_rec.get("valid", True):
        failures.append(f"full-system instanton defect is invalid: {full_rec.get('failure_reason') or 'invalid'}")
    if not skel_rec.get("valid", True):
        failures.append(f"skeleton instanton defect is invalid: {skel_rec.get('failure_reason') or 'invalid'}")

    full_below = full_rec.get("below_crossover")
    skel_below = skel_rec.get("below_crossover")
    between_crossovers = bool(full_below is not None and skel_below is not None and bool(full_below) != bool(skel_below))
    if between_crossovers:
        msg = ("temperature lies between the full-system and skeleton crossover temperatures; "
               "the steepest-descent kink mismatch makes the anchor uncontrolled")
        if allow_crossover_mismatch:
            warnings.append(msg + " (override enabled)")
        else:
            failures.append(msg)

    crossover_rel_diff = None
    cb_full = full_rec.get("crossover_beta")
    cb_skel = skel_rec.get("crossover_beta")
    if cb_full is not None and cb_skel is not None and float(cb_full) > 0:
        crossover_rel_diff = abs(float(cb_full) - float(cb_skel)) / float(cb_full)
        if crossover_rel_diff > float(crossover_rtol):
            warnings.append(
                f"skeleton crossover beta {float(cb_skel):.6g} differs from full crossover beta "
                f"{float(cb_full):.6g} by {crossover_rel_diff:.1%}"
            )

    a_full = float(full_rec["log_factor"]) if full_rec.get("valid", True) and full_rec.get("log_factor") is not None else float("nan")
    a_skel = float(skel_rec["log_factor"]) if skel_rec.get("valid", True) and skel_rec.get("log_factor") is not None else float("nan")
    log_shift = a_full - a_skel
    shift_factor = float(np.exp(log_shift)) if np.isfinite(log_shift) else float("nan")
    valid = bool(not failures and np.isfinite(log_shift) and np.isfinite(shift_factor) and shift_factor > 0)

    naive_log_shift = None
    naive_anchored_factor = None
    if naive_rec is not None and naive_rec.get("valid", True) and np.isfinite(a_full):
        a_naive = float(naive_rec["log_factor"])
        if np.isfinite(a_naive):
            naive_log_shift = float(a_full - a_naive)
            naive_anchored_factor = float(correction.factor * np.exp(naive_log_shift))

    anchor_summary = {
        "kind": "skeleton",
        "valid": valid,
        "full_log_defect": a_full if np.isfinite(a_full) else None,
        "skeleton_log_defect": a_skel if np.isfinite(a_skel) else None,
        "log_shift": log_shift if np.isfinite(log_shift) else None,
        "shift_factor": shift_factor if np.isfinite(shift_factor) else None,
        "between_crossovers": between_crossovers,
        "crossover_relative_difference": crossover_rel_diff,
        "naive_log_shift": naive_log_shift,
        "warnings": warnings,
    }
    diagnostics = dict(correction.diagnostics)
    diagnostics["unanchored_factor"] = correction.factor
    diagnostics["unanchored_log_factor"] = correction.log_factor
    diagnostics["instanton_anchor"] = anchor_summary
    validity = dict(correction.validity)
    validity["valid"] = bool(correction.valid and valid)
    validity["failures"] = list(correction.validity.get("failures", [])) + failures
    anchored_factor = correction.factor * shift_factor if np.isfinite(shift_factor) else float("nan")
    anchored_correction = SubspaceCorrection(
        correction.beta,
        correction.dimension,
        anchored_factor,
        correction.log_standard_error,
        correction.quantum_rate,
        correction.rpmd_rate,
        correction.nbeads,
        correction.model,
        correction.quantum,
        correction.rpmd,
        diagnostics,
        valid,
        validity,
    )
    return _json_safe({
        "valid": valid,
        "failures": failures,
        "warnings": warnings,
        "beta": beta,
        "dimension": correction.dimension,
        "unanchored_factor": correction.factor,
        "unanchored_log_factor": correction.log_factor,
        "full_log_defect": a_full if np.isfinite(a_full) else None,
        "skeleton_log_defect": a_skel if np.isfinite(a_skel) else None,
        "log_shift": log_shift if np.isfinite(log_shift) else None,
        "shift_factor": shift_factor if np.isfinite(shift_factor) else None,
        "anchored_factor": anchored_correction.factor if valid else None,
        "anchored_log_factor": anchored_correction.log_factor if valid else None,
        "log_standard_error": correction.log_standard_error,
        "between_crossovers": between_crossovers,
        "crossover_relative_difference": crossover_rel_diff,
        "naive_log_shift": naive_log_shift,
        "naive_anchored_factor": naive_anchored_factor,
        "full_defect": full_rec,
        "skeleton_defect": skel_rec,
        "naive_defect": naive_rec,
        "anchored_correction": anchored_correction,
        "approximation": "skeleton instanton anchor: ln C_S + ln eps_SD(H) - ln eps_SD(S^0)",
        "exact_general_rate_claim": False,
    }) | {"anchored_correction": anchored_correction}


__all__ = [
    "SubspaceCorrection", "apply_subspace_correction", "correct_rate_run", "full_instanton_defect",
    "harmonic_bead_factor", "instanton_anchor", "nested_subspace_sensitivity", "subspace_correction",
    "subspace_instanton_defect", "validity_gates", "validity_policy",
]
