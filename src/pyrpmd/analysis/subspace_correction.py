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
                   "max_outside_fraction": 0.01}


def validity_policy(policy: Mapping[str, Any] | None = None) -> dict:
    """Complete and check a validity policy (see :data:`VALIDITY_POLICY`)."""
    merged = dict(VALIDITY_POLICY)
    for key, value in dict(policy or {}).items():
        if key in merged and value is not None:
            merged[key] = value
    for key in ("unconverged_path", "unstable_bath", "outside_grid"):
        if merged[key] not in ("reject", "allow"):
            raise ValueError(f"validity policy {key} must be 'reject' or 'allow'")
    limit = float(merged["max_outside_fraction"])
    if not np.isfinite(limit) or not 0 <= limit <= 1:
        raise ValueError("max_outside_fraction must lie in [0, 1]")
    merged["max_outside_fraction"] = limit
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

    ``construction`` is the model provenance.  Three gates are evaluated when
    their evidence is present:

    ``path``   both continuations of an assembled reaction path ended at an
               accepted terminus (``path_termination``, set by the pipeline):
               an asymptote, or on the product side a well deep enough below
               the barrier to commit products;
    ``bath``   the normal-plane Hessian of a path projection is stable
               (``bath_stable``);
    ``grid``   the ring-polymer estimator did not sample the constant
               continuation outside a gridded ``V_S`` (``grid_exit`` of the
               RPMD diagnostics; absent before the rates are solved).

    The policy stored in the provenance (``validity_policy``) is used unless
    ``policy`` is given.  Returns ``{"valid", "failures", "overrides",
    "gates", "policy"}``; ``failures`` lists the gates that reject.
    """
    construction = dict(construction or {})
    policy = validity_policy(policy if policy is not None else construction.get("validity_policy"))
    gates = []
    termination = construction.get("path_termination")
    if termination is not None:
        ends = {side: dict(termination.get(side) or {}) for side in ("reactant", "product")}
        bad = [f"{side}: {end.get('status', 'no termination record')}"
               + (f" -- {end['reason']}" if end.get("reason") else "")
               for side, end in ends.items() if not end.get("converged")]
        gates.append(_gate("path", not bad, policy["unconverged_path"],
                           "reaction path did not reach an accepted terminus (" + "; ".join(bad) + ")",
                           {side: _path_evidence(end) for side, end in ends.items()}))
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
    exits = (rpmd_diagnostics or {}).get("grid_exit")
    if exits is not None:
        limit = policy["max_outside_fraction"]
        worst = float(exits["estimator_outside_fraction"])
        gates.append(_gate("grid", worst <= limit, policy["outside_grid"],
                           f"{worst:.3g} of the ring-polymer estimator sampled V_S outside its grid "
                           f"(limit {limit:.3g})",
                           {k: exits.get(k) for k in (
                               "ti_sample_outside_fraction", "surface_sample_outside_fraction",
                               "recrossing_exited_fraction", "recrossing_returned_fraction",
                               "recrossing_uphill_exit_fraction", "recrossing_contaminated_flux_fraction",
                               "recrossing_max_edge_slope", "estimator_outside_fraction")}
                           | {"max_outside_fraction": limit}))
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
            "approximation": "multiplicative subspace defect: k_RPMD(full) * k_QM(S) / k_RPMD(S)",
            "exact_general_rate_claim": False,
            "physical_converged": False,
        }
        if self.quantum is not None:
            record["quantum"] = self.quantum.to_dict()
        if self.rpmd is not None:
            record["rpmd"] = self.rpmd.to_dict()
        return _json_safe(record)

    def write(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
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
        return cls.from_dict(json.loads(Path(path).read_text()))


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
) -> dict:
    """Multiply a full-dimensional RPMD rate by ``C_S``.

    Relative errors are combined in quadrature assuming the full RPMD run and
    the subspace calculation are statistically independent.  ``beta`` (when
    given) must match the correction's temperature.  ``bead_factor`` (from
    :func:`harmonic_bead_factor`) divides the RPMD rate first, removing the
    environment's finite-bead error that an adiabatic ``H_S`` does not carry.
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
    corrected = rate * correction.factor / bead
    if standard_error is not None:
        standard_error = float(standard_error)
        if not np.isfinite(standard_error) or standard_error < 0:
            raise ValueError("standard_error must be finite and nonnegative")
        relative = float(np.hypot(standard_error / rate, correction.log_standard_error)) if rate > 0 else None
    return {
        "rpmd_rate": rate,
        "rpmd_rate_standard_error": standard_error,
        "correction_factor": correction.factor,
        "correction_log_standard_error": correction.log_standard_error,
        "bead_factor": None if bead_factor is None else bead,
        "corrected_rate": corrected,
        "corrected_rate_standard_error": None if relative is None else corrected * relative,
        "corrected_relative_standard_error": relative,
        "subspace_dimension": correction.dimension,
        "beta": correction.beta,
        "approximation": "multiplicative subspace defect",
        "physical_converged": False,
    }


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
            previous = json.loads(target.read_text())
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
        "physical_converged": False,
    }
    result = _json_safe(result)
    write_json(target, result)
    return result


__all__ = [
    "SubspaceCorrection", "apply_subspace_correction", "correct_rate_run", "harmonic_bead_factor",
    "nested_subspace_sensitivity", "subspace_correction", "validity_gates", "validity_policy",
]
