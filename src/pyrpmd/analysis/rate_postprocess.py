"""Static FES fluxes and channel-resolved signed-flux postprocessing.

All energies are hartree, beta is inverse hartree, masses are electron masses,
and time is atomic time. Bimolecular rates are bohr**3 / atomic time internally
and cm**3 molecule**-1 s**-1 on export. A coordinate-density FES and a FES
sampled in a rate-flux measure are different objects and are never interchanged.

These functions do not establish source convergence, asymptotic completeness,
or a physical product partition. Those are properties of the supplied sampling
and basin definitions. Every result therefore keeps ``converged=False``.
"""

from __future__ import annotations

from functools import partial

import numpy as np
from scipy.special import logsumexp

from ..constants import AU_RATE_TO_CM3_S
from ..validation import checked_integer as _integer
from ..validation import finite_positive
from .fes import _block_resample, statistical_inefficiency
from .recrossing import _plateau_diagnostics


def _real_array(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc


_positive = partial(
    finite_positive,
    shape="zero_dim",
    type_message="{name} must be a finite scalar",
)


class _Grid:
    def __init__(self, free_energy, edges, mask, beta):
        self.beta = _positive(beta, "beta")
        self.f = _real_array(free_energy, "free_energy")
        if self.f.ndim != 2 or any(n < 1 for n in self.f.shape):
            raise ValueError("free_energy must be a nonempty two-dimensional array")
        if not isinstance(edges, (list, tuple)) or len(edges) != 2:
            raise ValueError("edges must contain the two histogram edge arrays")
        self.edges = [_real_array(e, f"edges[{i}]") for i, e in enumerate(edges)]
        for i, e in enumerate(self.edges):
            if (e.ndim != 1 or len(e) != self.f.shape[i] + 1
                    or not np.isfinite(e).all() or not np.isfinite(np.diff(e)).all()
                    or np.any(np.diff(e) <= 0)):
                raise ValueError("edges must be finite, strictly increasing and match free_energy")
        if mask is None:
            supplied = np.zeros(self.f.shape, dtype=bool)
        else:
            supplied = np.asarray(mask)
            if supplied.dtype.kind != "b" or supplied.shape != self.f.shape:
                raise ValueError("mask must be a boolean array matching free_energy")
        self.mask = supplied | ~np.isfinite(self.f)
        if np.all(self.mask):
            raise ValueError("free_energy has no supported bins")
        self.mid = [(e[:-1] + e[1:]) / 2 for e in self.edges]
        self.reference = float(np.min(self.f[~self.mask]))
        self.log_weight = np.zeros(self.f.shape)
        with np.errstate(over="ignore", invalid="ignore"):
            self.log_weight[~self.mask] = -self.beta * (self.f[~self.mask] - self.reference)
        if not np.isfinite(self.log_weight[~self.mask]).all():
            raise ValueError("beta times the supported free-energy range is not finite")

    def axis_weights(self, axis, points):
        points = np.asarray(points)
        e, mid = self.edges[axis], self.mid[axis]
        if np.any(points < e[0]) or np.any(points > e[-1]):
            raise ValueError("curve/slice extends outside the supplied FES domain")
        if len(mid) == 1:
            return np.zeros(points.shape, int), np.zeros(points.shape, int), np.zeros(points.shape)
        hi = np.clip(np.searchsorted(mid, points, side="right"), 1, len(mid) - 1)
        lo = hi - 1
        gap = mid[hi] - mid[lo]
        fraction = np.clip((points - mid[lo]) / gap, 0.0, 1.0)
        # A decimal slice such as 1.0 and its histogram centre computed as
        # (left+right)/2 may differ by an ULP. Such arithmetic noise must not
        # create a spurious nonzero weight on a masked adjacent row. Cap the
        # tolerance by a tiny fraction of the actual cell spacing so that
        # closely spaced, physically distinct coordinates remain distinct.
        scale = np.maximum(1.0, np.maximum(np.abs(points), np.maximum(np.abs(mid[lo]), np.abs(mid[hi]))))
        tolerance = np.minimum(8*np.finfo(float).eps*scale, 1e-8*gap)
        fraction = np.where(np.abs(points-mid[lo]) <= tolerance, 0.0, fraction)
        fraction = np.where(np.abs(points-mid[hi]) <= tolerance, 1.0, fraction)
        return lo, hi, fraction

    def interpolate(self, points):
        """Bilinear F interpolation, constant in the exterior half bins.

        Only nonzero interpolation weights request support. Thus a curve on a
        supported row does not spuriously depend on an unused neighboring row.
        """
        a, b, u = self.axis_weights(0, points[:, 0])
        c, d, v = self.axis_weights(1, points[:, 1])
        result = np.zeros(len(points))
        for ix, iy, weight in ((a, c, (1-u)*(1-v)), (a, d, (1-u)*v),
                               (b, c, u*(1-v)), (b, d, u*v)):
            used = weight > 0
            if np.any(self.mask[ix[used], iy[used]]):
                raise ValueError("curve/slice touches masked or unsupported FES interpolation bins")
            result[used] += weight[used] * self.log_weight[ix[used], iy[used]]
        return result

    def slice_log_integral(self, point):
        if isinstance(point, (bool, np.bool_)) or np.ndim(point) != 0:
            raise ValueError("slice coordinate must be a finite scalar")
        point = float(_real_array(point, "slice coordinate"))
        if not np.isfinite(point):
            raise ValueError("slice coordinate must be finite")
        # Transverse integration uses histogram densities and exact bin widths,
        # not interpolation across the transverse bins. At an x-bin centre this
        # is precisely the 2D histogram's marginal density.
        points = np.column_stack([np.full(len(self.mid[1]), point), self.mid[1]])
        log_values = self.interpolate(points)
        return float(logsumexp(log_values + np.log(np.diff(self.edges[1]))))


def _speed_model(normal_speed, inverse_mass, curve):
    if (normal_speed is None) == (inverse_mass is None):
        raise ValueError("specify exactly one of normal_speed or inverse_mass")
    if inverse_mass is not None:
        return "inverse_mass", _positive(inverse_mass, "inverse_mass")
    if callable(normal_speed):
        return "callable", normal_speed
    speed = _real_array(normal_speed, "normal_speed")
    if speed.ndim == 0:
        return "scalar", _positive(speed, "normal_speed")
    if speed.shape != (len(curve),) or not np.isfinite(speed).all() or np.any(speed <= 0):
        raise ValueError("normal_speed must be positive at each curve vertex")
    return "vertices", speed


def _curve_integral(grid, curve, normal_speed, inverse_mass, quadrature_order):
    curve = _real_array(curve, "curve")
    if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2 or not np.isfinite(curve).all():
        raise ValueError("curve must contain at least two finite [s0,s1] vertices")
    for axis in range(2):
        grid.axis_weights(axis, curve[:, axis])
    delta = np.diff(curve, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths <= 0) or not np.isfinite(lengths).all():
        raise ValueError("curve contains a zero-length or invalid segment")
    order = _integer(quadrature_order, "quadrature_order", 2)
    if order > 128:
        raise ValueError("quadrature_order must be <=128")
    kind, speed = _speed_model(normal_speed, inverse_mass, curve)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    terms = []
    panels = 0
    for segment, (origin, displacement, length) in enumerate(zip(curve[:-1], delta, lengths)):
        normal = np.array([-displacement[1], displacement[0]]) / length
        # Split at every interpolation-cell boundary. A narrow unsupported cell
        # cannot be jumped over by a coarse user polyline or quadrature node set.
        cuts = [0.0, 1.0]
        for axis in range(2):
            if displacement[axis] != 0:
                crossings = (grid.mid[axis] - origin[axis]) / displacement[axis]
                cuts.extend(crossings[(crossings > 0) & (crossings < 1)].tolist())
        cuts = np.unique(cuts)
        for left, right in zip(cuts[:-1], cuts[1:]):
            t = (left+right)/2 + (right-left)/2 * nodes
            points = origin[None, :] + t[:, None] * displacement[None, :]
            log_density = grid.interpolate(points)
            if kind == "inverse_mass":
                velocities = np.full(order, np.sqrt(speed / (2*np.pi*grid.beta)))
            elif kind == "scalar":
                velocities = np.full(order, speed)
            elif kind == "vertices":
                velocities = (1-t)*speed[segment] + t*speed[segment+1]
            else:
                velocities = _real_array(speed(points, np.broadcast_to(normal, points.shape)), "normal_speed output")
                if velocities.ndim == 0:
                    velocities = np.full(order, velocities)
                if velocities.shape != (order,):
                    raise ValueError("normal_speed callable must return one velocity per point")
            if not np.isfinite(velocities).all() or np.any(velocities <= 0):
                raise ValueError("normal_speed must be positive and finite at every quadrature point")
            terms.extend((log_density + np.log(velocities) + np.log(weights)
                          + np.log(length*(right-left)/2)).tolist())
            panels += 1
    return float(logsumexp(terms)), {
        "curve_vertices": len(curve), "curve_length": float(lengths.sum()),
        "quadrature_panels": panels, "quadrature_order": order,
        "interpolation": "bilinear free energy at bin centres; constant exterior half bins",
        "velocity_model": kind,
    }


def _absolute_log(grid, shifted):
    with np.errstate(over="ignore", invalid="ignore"):
        value = shifted - grid.beta * grid.reference
    return float(value) if np.isfinite(value) else None


def _rate_result(log_rate, *, measure, normalization_kind, **details):
    if not np.isfinite(log_rate):
        raise ValueError("normalized log rate must be finite")
    if log_rate > np.log(np.finfo(float).max) - max(0, np.log(AU_RATE_TO_CM3_S)):
        raise ValueError("normalized rate overflows floating-point representation")
    with np.errstate(under="ignore"):
        rate = float(np.exp(log_rate))
    return {
        "schema": "pyrpmd.static_rate_postprocess.v1",
        "status": "static_tst_estimate", "measure": measure,
        "normalization_kind": normalization_kind,
        "k_tst_au": rate, "k_tst_cm3_s": rate*AU_RATE_TO_CM3_S,
        "log_k_tst_au": float(log_rate),
        "rate_unit": "cm^3 molecule^-1 s^-1", "energy_unit": "hartree",
        "time_unit": "atomic_time", "support_complete": True,
        "underflow": bool(rate == 0), "converged": False,
        "physical_scope": "Statistical TST flux in the supplied finite CV domain and normalization; source, transverse-tail and dynamical convergence are not established.",
        **details,
    }


def surface_integral_2d(free_energy, edges, *, beta, curve, normal_speed=None,
                        inverse_mass=None, mask=None, measure="cv_density",
                        quadrature_order=8):
    """Integrate ``exp(-beta F) <v_n theta(v_n)> dl`` along a polyline.

    The two CV axes must have compatible coordinate scales (or an explicitly
    chosen nondimensionalization); their inverse-mass/velocity units must match.
    ``normal_speed`` is a positive scalar, values at curve vertices, or a
    callable ``normal_speed(points[N,2], normals[N,2]) -> speed[N]``. It is the
    conditional mean positive flux velocity, not the mean conditional on v>0.
    The alternative constant ``inverse_mass`` gives sqrt(inverse_mass/(2pi beta)).

    The absolute integral depends on the arbitrary F zero. Use the normalized
    rate functions below for physically meaningful, gauge-invariant results.
    """
    if measure != "cv_density":
        raise ValueError("general curves require measure='cv_density'; a rate_flux FES is tied to its original flux axis")
    grid = _Grid(free_energy, edges, mask, beta)
    shifted, details = _curve_integral(grid, curve, normal_speed, inverse_mass, quadrature_order)
    return {
        "log_integral": _absolute_log(grid, shifted),
        "log_integral_shifted": shifted, "energy_reference_hartree": grid.reference,
        "measure": measure, "support_complete": True, "converged": False,
        "energy_unit": "hartree", "time_unit": "atomic_time", **details,
    }


def surface_rate_2d(free_energy, edges, *, beta, curve, normal_speed=None,
                    inverse_mass=None, mask=None, measure="cv_density",
                    reactant_mask=None, reactant_volume_au=None,
                    reference_curve=None, reference_normal_speed=None,
                    reference_inverse_mass=None, reference_rate_au=None,
                    quadrature_order=8):
    """Bimolecular TST rate from a CV-density surface and explicit normalization.

    Supply either a free-reactant basin mask plus its accessible relative
    translation volume (bohr^3), or a complete reference curve with known
    bimolecular positive flux (bohr^3/atomic_time). A contact-complex basin is
    not a free-reactant normalization. Reference and target must be complete
    flux sections of the same sampled ensemble; the function cannot prove this.
    """
    if measure != "cv_density":
        raise ValueError("surface_rate_2d requires measure='cv_density'")
    basin_mode = reactant_mask is not None or reactant_volume_au is not None
    reference_mode = any(x is not None for x in (
        reference_curve, reference_normal_speed, reference_inverse_mass, reference_rate_au))
    if basin_mode == reference_mode:
        raise ValueError("specify exactly one normalization: free-reactant basin/volume or reference curve/rate")
    grid = _Grid(free_energy, edges, mask, beta)
    target, details = _curve_integral(grid, curve, normal_speed, inverse_mass, quadrature_order)
    common = {"log_surface_integral": _absolute_log(grid, target),
              "log_surface_integral_shifted": target, "energy_reference_hartree": grid.reference,
              "surface_diagnostics": details}
    if basin_mode:
        volume = _positive(reactant_volume_au, "reactant_volume_au")
        region = np.asarray(reactant_mask)
        if region.shape != grid.f.shape or region.dtype.kind != "b" or not region.any():
            raise ValueError("reactant_mask must be a nonempty boolean region matching free_energy")
        if np.any(grid.mask & region):
            raise ValueError("reactant normalization touches masked or unsupported bins")
        areas = np.diff(grid.edges[0])[:, None] * np.diff(grid.edges[1])[None, :]
        normalizer = float(logsumexp(grid.log_weight[region] + np.log(areas[region])))
        return _rate_result(np.log(volume)+target-normalizer, measure=measure,
            normalization_kind="free_reactant_volume", reactant_volume_au=volume,
            log_reactant_partition=_absolute_log(grid, normalizer), **common)
    if reference_curve is None or reference_rate_au is None:
        raise ValueError("reference normalization requires reference_curve and reference_rate_au")
    reference_rate = _positive(reference_rate_au, "reference_rate_au")
    if reference_normal_speed is None and reference_inverse_mass is None:
        if inverse_mass is not None:
            reference_inverse_mass = inverse_mass
        elif callable(normal_speed) or np.ndim(normal_speed) == 0:
            reference_normal_speed = normal_speed
        else:
            raise ValueError("reference curve needs its own vertex normal_speed values")
    reference, ref_details = _curve_integral(grid, reference_curve, reference_normal_speed,
                                            reference_inverse_mass, quadrature_order)
    return _rate_result(np.log(reference_rate)+target-reference, measure=measure,
        normalization_kind="reference_flux", reference_rate_au=reference_rate,
        log_reference_integral=_absolute_log(grid, reference), log_flux_ratio=target-reference,
        reference_diagnostics=ref_details, **common)


def flux_slice_rate_2d(free_energy, edges, *, beta, surface, reference,
                       reference_rate_au, mask=None, measure="rate_flux", flux_axis=0):
    """Rate-flux FES ratio for complete constant-s0 slices only.

    The sampled measure must already contain the positive normal flux for s0.
    No extra thermal velocity or mass factor is inserted. Rotating this surface
    or treating axis 1 as the flux axis is invalid. Transverse integration is
    the histogram sum over *all* s1 bins with their individual widths. F is
    linearly interpolated only along s0, with constant exterior half bins.
    """
    if measure != "rate_flux":
        raise ValueError("flux_slice_rate_2d requires measure='rate_flux'")
    if (isinstance(flux_axis, (bool, np.bool_))
            or not isinstance(flux_axis, (int, np.integer)) or flux_axis != 0):
        raise ValueError("rate_flux slices support flux_axis=0 only; the sampled flux measure cannot be rotated")
    rate_ref = _positive(reference_rate_au, "reference_rate_au")
    grid = _Grid(free_energy, edges, mask, beta)
    target = grid.slice_log_integral(surface)
    source = grid.slice_log_integral(reference)
    return _rate_result(np.log(rate_ref)+target-source, measure=measure,
        normalization_kind="reference_flux", reference_rate_au=rate_ref,
        surface=float(surface), reference=float(reference), flux_axis=0,
        log_surface_integral=_absolute_log(grid, target),
        log_reference_integral=_absolute_log(grid, source), log_flux_ratio=target-source,
        energy_reference_hartree=grid.reference,
        integration="full transverse histogram-density sum; axial free-energy interpolation",
        transverse_domain=[float(grid.edges[1][0]), float(grid.edges[1][-1])])


def channel_rates(k_tst_au, time_au, parent_numerator, parent_denominator,
                  parent_ids, channel_names, *, plateau_fraction=0.25,
                  plateau_tolerance=0.1, bootstrap=200, seed=0,
                  source_identity=None, flux_identity=None):
    """Postprocess a common entrance's signed channel fluxes jointly.

    numerator has shape [parent,time,channel], denominator [parent], and IDs
    [parent,2] contain unique integer (chain,parent) pairs. Child pairs sharing
    a parent must already be aggregated. Numerators are signed flux-side sums,
    never positive-trajectory outcome counts. All channels use identical
    bootstrap indices, the same denominator and a common conservative block
    length per chain. No negative flux or unphysical branch estimate is clipped.

    Rates are conditional on the supplied static TST factor: its uncertainty
    is not inferred from these records. Plateau detection is a finite-window
    diagnostic, not proof of physical convergence or product completeness.
    """
    static = _positive(k_tst_au, "k_tst_au", allow_zero=True)
    times = _real_array(time_au, "time_au")
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or times[0] < 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("time_au must be finite, nonnegative and strictly increasing")
    num = _real_array(parent_numerator, "parent_numerator")
    den = _real_array(parent_denominator, "parent_denominator")
    names = list(channel_names) if not isinstance(channel_names, str) else []
    if (not names or any(not isinstance(s, str) or not s for s in names)
            or len(set(names)) != len(names)):
        raise ValueError("channel_names must be unique nonempty strings")
    if num.ndim != 3 or num.shape[1:] != (len(times), len(names)) or num.shape[0] < 1:
        raise ValueError("parent_numerator must have shape [parent,time,channel]")
    if (den.shape != (len(num),) or not np.isfinite(num).all()
            or not np.isfinite(den).all() or np.any(den < 0)
            or not np.isfinite(den.sum()) or den.sum() <= 0):
        raise ValueError("finite signed numerators and nonnegative denominators with positive total are required")
    if np.any(num[den == 0] != 0):
        raise ValueError("a zero-denominator parent cannot carry nonzero signed flux")
    # A pair contributes at most its positive weight to an individual side
    # function or to the union of mutually exclusive product side functions.
    tolerance = 1e-10 * den[:, None]
    if (np.any(np.abs(num) > den[:, None, None] + tolerance[:, :, None])
            or np.any(np.abs(num.sum(axis=2)) > den[:, None] + tolerance)):
        raise ValueError("signed channel flux exceeds its parent denominator or channels are not mutually exclusive")
    ids = np.asarray(parent_ids)
    if (ids.shape != (len(num), 2) or ids.dtype.kind not in "iu" or np.any(ids < 0)
            or len({tuple(row) for row in ids.tolist()}) != len(num)):
        raise ValueError("parent_ids must be unique nonnegative integer (chain,parent) pairs")
    if (source_identity is None) != (flux_identity is None):
        raise ValueError("source_identity and flux_identity must be supplied together")
    if source_identity is not None:
        if (not isinstance(source_identity, str) or not source_identity
                or not isinstance(flux_identity, str) or source_identity != flux_identity):
            raise ValueError("static surface identity and signed flux identity do not match")
    fraction = _positive(plateau_fraction, "plateau_fraction")
    if fraction > 1:
        raise ValueError("plateau_fraction must be <=1")
    tolerance_fraction = _positive(plateau_tolerance, "plateau_tolerance")
    count = _integer(bootstrap, "bootstrap", 0)
    seed = _integer(seed, "seed", 0)
    order = np.lexsort((ids[:, 1], ids[:, 0]))
    ids, num, den = ids[order], num[order], den[order]
    curve = num.sum(axis=0) / den.sum()
    tail = min(len(times), max(2, int(np.ceil(len(times)*fraction))))
    start = len(times)-tail
    middle = start + tail//2
    estimate = curve[start:].mean(axis=0)
    total_curve = curve.sum(axis=1)
    total = float(estimate.sum())
    drift = curve[middle:].mean(axis=0)-curve[start:middle].mean(axis=0)
    chains = {int(c): np.flatnonzero(ids[:, 0] == c) for c in np.unique(ids[:, 0])}
    blocks, effective = {}, {}
    for c, ix in chains.items():
        observable = np.divide(num[ix, start:].mean(axis=1), den[ix, None],
                               out=np.zeros((len(ix), len(names))), where=den[ix, None] > 0)
        observables = np.column_stack([observable, observable.sum(axis=1)])
        # An exactly conserved total or an identically zero channel carries no
        # correlation information; it must not force every other channel's
        # block to the entire chain. If all are constant, mixing is unproven.
        varying = [col for col in observables.T if np.ptp(col) > 1e-14]
        g = max((statistical_inefficiency(col) for col in varying), default=float(len(ix)))
        blocks[c] = max(1, int(np.ceil(g)))
        effective[c] = float(np.count_nonzero(den[ix] > 0)/blocks[c])
    independent = sum(np.any(den[ix] > 0) for ix in chains.values())
    scope = ("independent_parent_chains" if independent >= 2 else
             "conditional_on_single_chain_mixing" if sum(effective.values()) >= 2 else "unavailable")
    rng = np.random.default_rng(seed)
    boot_curves = []
    for _ in range(count):
        selected = []
        for c in rng.choice(list(chains), size=len(chains), replace=True):
            selected.extend(_block_resample(chains[int(c)], blocks[int(c)], rng).tolist())
        selected = np.asarray(selected, int)
        denominator = den[selected].sum()
        if denominator > 0:
            boot_curves.append(num[selected].sum(axis=0)/denominator)
    boot_curves = np.asarray(boot_curves).reshape((-1, len(times), len(names)))
    draws = boot_curves[:, start:].mean(axis=1)
    uncertainty_reasons = []
    if scope == "unavailable":
        uncertainty_reasons.append("fewer_than_two_effective_parent_blocks_or_independent_chains")
    if len(draws) < 2:
        uncertainty_reasons.append("fewer_than_two_valid_bootstrap_draws")
    if len(draws) != count:
        uncertainty_reasons.append("bootstrap_contains_zero_flux_draws")
    available = not uncertainty_reasons
    error = draws.std(axis=0, ddof=1) if available else None
    curve_error = boot_curves.std(axis=0, ddof=1) if available else None
    drift_draws = (boot_curves[:, middle:].mean(axis=1)
                   - boot_curves[:, start:middle].mean(axis=1))
    drift_error = drift_draws.std(axis=0, ddof=1) if available else np.zeros(len(names))
    # Both each channel and their union must have reached a plateau. Stability
    # of the total alone cannot establish that branching has stopped evolving.
    # Reuse the scalar rate diagnostic: equal half-means do not rule out an
    # oscillating tail, and larger uncertainty must not relax the tolerance.
    channel_diagnostics = [
        _plateau_diagnostics(
            curve[:, j], start, float(estimate[j]), float(drift[j]),
            float(drift_error[j]) if available else None,
            boot_curves[:, :, j], available, tolerance_fraction,
        )
        for j in range(len(names))
    ]
    channel_plateaus = np.asarray([item["passed"] for item in channel_diagnostics], bool)
    total_drift = float(drift.sum())
    total_drift_error = float(drift_draws.sum(axis=1).std(ddof=1)) if available else None
    total_diagnostics = _plateau_diagnostics(
        total_curve, start, total, total_drift, total_drift_error,
        boot_curves.sum(axis=2), available, tolerance_fraction,
    )
    total_plateau = total_diagnostics["passed"]
    plateau = bool(channel_plateaus.all() and total_plateau)
    reasons = []
    if np.any(estimate < 0):
        reasons.append("negative_channel_net_flux")
    if total <= 0:
        reasons.append("nonpositive_total_net_flux")
    if static == 0:
        reasons.append("zero_static_rate")
    if not plateau:
        reasons.append("channel_or_total_plateau_not_established")
    valid = not reasons
    branch = estimate/total if total > 0 else None
    branch_draws = np.full(draws.shape, np.nan)
    positive_totals = draws.sum(axis=1) > 0
    branch_draws[positive_totals] = draws[positive_totals]/draws[positive_totals].sum(axis=1)[:, None]
    branch_error = branch_draws.std(axis=0, ddof=1) if available and positive_totals.all() else None
    rates = static*estimate
    rate_error = static*error if error is not None else None
    return {
        "schema": "pyrpmd.channel_rate_postprocess.v1", "status": "conditional_plateau_estimate" if valid else "pilot_requires_convergence",
        "valid": valid,
        "channel_names": names, "time_au": times, "kappa_curve": curve,
        "kappa": estimate, "kappa_estimates": estimate, "total_kappa": total,
        "total_kappa_curve": total_curve, "kappa_standard_error": error,
        "kappa_curve_standard_error": curve_error, "bootstrap_kappa": draws,
        "bootstrap_branching_fractions": branch_draws,
        "channel_rate_estimates_au": rates, "channel_rate_estimates_cm3_s": rates*AU_RATE_TO_CM3_S,
        "channel_rates_au": rates if valid else None,
        "channel_rates_cm3_s": rates*AU_RATE_TO_CM3_S if valid else None,
        "total_rate_estimate_au": float(rates.sum()),
        "total_rate_estimate_cm3_s": float(rates.sum()*AU_RATE_TO_CM3_S),
        "rate_estimate_cm3_s": float(rates.sum()*AU_RATE_TO_CM3_S),
        "total_rate_au": float(rates.sum()) if valid else None,
        "total_rate_cm3_s": float(rates.sum()*AU_RATE_TO_CM3_S) if valid else None,
        "branching_fraction_estimates": branch,
        "branching_fractions": branch if valid else None,
        "branching_fraction_standard_error": branch_error,
        "channel_rate_standard_error_au": rate_error,
        "channel_rate_standard_error_cm3_s": rate_error*AU_RATE_TO_CM3_S if rate_error is not None else None,
        "plateau_detected": plateau, "channel_plateau_detected": channel_plateaus,
        "channel_plateau_diagnostics": dict(zip(names, channel_diagnostics)),
        "total_plateau_diagnostics": total_diagnostics,
        "plateau_start_index": start, "plateau_start_au": float(times[start]),
        "plateau_drift": drift, "plateau_drift_standard_error": drift_error if available else None,
        "total_plateau_drift": total_drift, "parent_ids": ids,
        "parent_configurations": len(ids), "independent_parent_chains": int(independent),
        "parent_block_lengths": {str(c): b for c, b in blocks.items()},
        "rate_uncertainty": {
            "available": available, "reasons": uncertainty_reasons, "scope": scope,
            "conditional_quantity": "joint channel recrossing uncertainty with the common static TST rate held fixed",
            "static_uncertainty_included": False, "joint_channel_resampling": True,
            "bootstrap_requested": count, "bootstrap_valid": len(draws),
            "bootstrap_nonpositive_total": int(np.count_nonzero(~positive_totals)),
            "bootstrap_negative_channel_draws": int(np.count_nonzero(np.any(draws < 0, axis=1))),
            "effective_parent_blocks": {str(c): n for c, n in effective.items()},
        },
        "reasons": reasons, "identity_checked": source_identity is not None,
        "source_identity": source_identity, "rate_unit": "cm^3 molecule^-1 s^-1",
        "time_unit": "atomic_time", "converged": False,
        "physical_scope": "Conditional signed-flux plateau estimate for the supplied common entrance and product partition; source sampling, energy conservation and product completeness must be checked separately.",
    }
