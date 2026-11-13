"""Independent finite-bead H3 dividing-surface and reactant partition ratios.

This is equilibrium benchmark tooling, not a real-time RPMD propagator. At the
centroid transition surface x=y=r, z=r*t, its Cartesian coarea/flux shape measure
is r**4*t*sqrt(2+t*t) dr dt (1<=t<=2). The isolated reactant measure is r**2 dr.
Global rotations, equivalent-channel and thermal-wavelength constants cancel
within each quantum/classical ratio. Zero-centroid Jacobi spring priors are
normalized; the beadwise total COM has been integrated out.

The auxiliary reference potential evaluates the centroid PES at max(r,a), where
a is the minimum allowed bead pair distance. It has full centroid support down
to r=0, including an analytically integrated repulsive plateau. The target has
only the declared beadwise lower bound and centroid upper bound. An independent
reference-law/free-ring experiment estimates the nontrivial lambda=0 domain
probability. That probability and Z_ref/Z_classical are never silently set to 1.

U_lambda=(1-lambda)*V_ref+lambda*mean(V_beads). Adjacent-lambda Bennett acceptance
ratios estimate log Z_1/Z_0; a thermodynamic-integration curve is a cross-check.
The primary uncertainty unit is an independent seed chain, not a saved sweep.
BAR reference: Bennett, J. Comput. Phys. 22, 245 (1976),
https://doi.org/10.1016/0021-9991(76)90078-4 . The equations are implemented here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.integrate import simpson, trapezoid
from scipy.optimize import brentq
from scipy.special import expit, logsumexp, roots_legendre
from scipy.stats import beta as beta_distribution

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.h3.classical_tst import MASS_AU
from benchmarks.h3.mixing_sampling import free_ring
from benchmarks.h3.pes import BKMP2Potential
from pyrpmd.analysis.fes import statistical_inefficiency
from pyrpmd.checkpoint import atomic_savez, load_checkpoint
from pyrpmd.constants import KB_AU
from pyrpmd.io import jsonable, write_json


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite_exp(value):
    """Keep log estimates usable even when their linear representation overflows."""
    return float(np.exp(value)) if value <= np.log(np.finfo(float).max) else None


def shape_log_weight(surface, radius, angle_ratio=None):
    """Coarea/flux density after factoring constant rotations and masses."""
    radius = np.asarray(radius, float)
    if surface not in ("ts", "reactant"):
        raise ValueError("surface must be ts or reactant")
    valid = np.isfinite(radius) & (radius > 0)
    result = np.full(radius.shape, -np.inf)
    if surface == "reactant":
        result[valid] = 2*np.log(radius[valid])
    else:
        ratio = np.broadcast_to(np.asarray(angle_ratio, float), radius.shape)
        valid &= np.isfinite(ratio) & (ratio >= 1) & (ratio <= 2)
        result[valid] = (4*np.log(radius[valid])+np.log(ratio[valid])
                         + .5*np.log(2+ratio[valid]**2))
    return result


def transition_centroids(radius, angle_ratio):
    """One orientation of the active surface r01=r12=r, r02=r*t."""
    radius, ratio = np.broadcast_arrays(radius, angle_ratio)
    cosine = 1-ratio*ratio/2
    q = np.zeros(radius.shape+(3, 3))
    q[..., 0, 0] = radius
    q[..., 2, 0] = radius*cosine
    q[..., 2, 1] = radius*np.sqrt(np.maximum(0., 1-cosine*cosine))
    return q


def _energies(provider, q):
    result = provider.evaluate(np.ascontiguousarray(q).reshape(-1, 3, 3))
    if np.any(result.status) or not np.isfinite(result.energy).all():
        raise ValueError("PES failed inside the declared physical partition domain")
    return np.asarray(result.energy).reshape(np.asarray(q).shape[:-2])


def diatom_energy(provider, radius):
    """Same 99-bohr equal-spectator definition as the classical TST oracle."""
    radius = np.asarray(radius, float)
    if np.any(radius >= 198):
        raise ValueError("diatom radius exceeds the finite-spectator construction")
    q = np.zeros(radius.shape+(3, 3))
    q[..., 0, 0] = -radius/2
    q[..., 1, 0] = radius/2
    q[..., 2, 2] = np.sqrt(99**2-(radius/2)**2)
    return _energies(provider, q)


def reference_energy(provider, surface, radius, angle_ratio, minimum):
    clamped = np.maximum(np.asarray(radius, float), minimum)
    if surface == "reactant":
        return diatom_energy(provider, clamped)
    return _energies(provider, transition_centroids(clamped, angle_ratio))


@dataclass
class ReferenceLaw:
    surface: str
    beta: float
    minimum: float
    maximum: float
    core_radius: np.ndarray
    core_ratio: np.ndarray
    core_log_weights: np.ndarray
    plateau_ratio: np.ndarray
    plateau_log_weights: np.ndarray
    radial_order: int
    angular_order: int

    @property
    def log_classical(self):
        return float(logsumexp(self.core_log_weights))

    @property
    def log_plateau(self):
        return float(logsumexp(self.plateau_log_weights))

    @property
    def log_reference(self):
        return float(np.logaddexp(self.log_classical, self.log_plateau))

    def sample(self, rng, count):
        """Quadrature-distributed core centroids; exact radial plateau draws.

        Domain probabilities have a separate quadrature error to check by
        refinement. The Markov chains themselves use continuous shape moves.
        """
        logs = np.r_[self.core_log_weights, self.plateau_log_weights]
        which = rng.choice(len(logs), size=count, p=np.exp(logs-self.log_reference))
        radii = np.r_[self.core_radius, np.full(len(self.plateau_ratio), self.minimum)][which]
        ratios = np.r_[self.core_ratio, self.plateau_ratio][which]
        plateau = which >= len(self.core_radius)
        power = 5 if self.surface == "ts" else 3
        uniform = np.maximum(rng.random(int(plateau.sum())), np.nextafter(0., 1.))
        radii[plateau] *= uniform**(1/power)
        return radii, ratios

    def report(self):
        plateau = self.log_plateau
        return {"surface": self.surface, "radial_order": self.radial_order,
                "angular_order": self.angular_order,
                "centroid_domain_bohr": [0., self.maximum],
                "classical_radial_domain_bohr": [self.minimum, self.maximum],
                "log_reference_shape_integral": self.log_reference,
                "log_classical_shape_integral": self.log_classical,
                "log_reference_over_classical": self.log_reference-self.log_classical,
                "log_plateau_shape_integral": plateau if np.isfinite(plateau) else None,
                "log_plateau_probability": plateau-self.log_reference if np.isfinite(plateau) else None,
                "plateau_radial_integration": "analytic r^5/5" if self.surface == "ts" else "analytic r^3/3",
                "reference_distribution_approximation": "Gauss-Legendre centroid quadrature; domain probability needs quadrature refinement"}


def reference_quadrature(provider, surface, beta, *, minimum=.5, maximum=8.,
                         radial_order=192, angular_order=96):
    if (surface not in ("ts", "reactant") or not np.isfinite([beta, minimum, maximum]).all()
            or beta <= 0 or not 0 <= minimum < maximum < 20
            or min(radial_order, angular_order) < 8):
        raise ValueError("invalid reference surface, beta, domain or quadrature order")
    node, weight = roots_legendre(radial_order)
    radius = minimum+(node+1)*(maximum-minimum)/2
    wr = weight*(maximum-minimum)/2
    if surface == "ts":
        node, weight = roots_legendre(angular_order)
        ratio, wt = 1.5+node/2, weight/2
        rr, tt = np.meshgrid(radius, ratio, indexing="ij")
        energy = reference_energy(provider, surface, rr, tt, minimum)
        core = (np.log(wr[:, None])+np.log(wt[None, :])
                + shape_log_weight(surface, rr, tt)-beta*energy)
        if minimum:
            plateau = (np.log(wt)+np.log(ratio)+.5*np.log(2+ratio*ratio)
                       +5*np.log(minimum)-np.log(5)
                       -beta*reference_energy(provider, surface, minimum*np.ones_like(ratio), ratio, minimum))
        else:
            ratio, plateau = np.empty(0), np.empty(0)
        return ReferenceLaw(surface, beta, minimum, maximum, rr.ravel(), tt.ravel(),
                            core.ravel(), ratio, plateau, radial_order, angular_order)
    energy = reference_energy(provider, surface, radius, np.ones_like(radius), minimum)
    core = np.log(wr)+2*np.log(radius)-beta*energy
    plateau = (np.array([3*np.log(minimum)-np.log(3)
                        -beta*float(reference_energy(provider, surface, np.array(minimum), None, minimum))])
               if minimum else np.empty(0))
    return ReferenceLaw(surface, beta, minimum, maximum, radius, np.ones_like(radius), core,
                        np.ones(len(plateau)), plateau, radial_order, angular_order)


def path_coordinates(surface, radius, ratio, eta_R, eta_r):
    """Build retained Jacobi paths and their beadwise domain diagnostic."""
    if surface == "reactant":
        relative = eta_r.copy()
        relative[..., 0] += radius[:, None]
        return None, relative, np.linalg.norm(relative, axis=-1).min(axis=1)
    q = transition_centroids(radius, ratio)
    center_R = q[:, 0]-(q[:, 1]+q[:, 2])/2
    center_r = q[:, 2]-q[:, 1]
    R, relative = center_R[:, None]+eta_R, center_r[:, None]+eta_r
    minimum = np.minimum(np.linalg.norm(relative, axis=-1),
                         np.minimum(np.linalg.norm(R+relative/2, axis=-1),
                                    np.linalg.norm(R-relative/2, axis=-1))).min(axis=1)
    return R, relative, minimum


def evaluate_paths(provider, reference, radius, ratio, eta_R, eta_r):
    shape = shape_log_weight(reference.surface, radius, ratio)
    R, relative, minimum = path_coordinates(reference.surface, radius, ratio, eta_R, eta_r)
    valid = np.isfinite(shape) & (radius <= reference.maximum) & (minimum >= reference.minimum)
    vref, energy = np.full(len(radius), np.inf), np.full(len(radius), np.inf)
    delta = np.full(len(radius), np.nan)
    if valid.any():
        vref[valid] = reference_energy(provider, reference.surface, radius[valid], ratio[valid], reference.minimum)
        if eta_r.shape[1] == 1:
            # With one bead, an accepted centroid is in the classical domain,
            # V_ref=V_centroid=V_bead. Enforce this exact identity without a
            # second, roundoff-different translated Cartesian PES evaluation.
            energy[valid] = vref[valid]
        elif reference.surface == "reactant":
            energy[valid] = diatom_energy(provider, np.linalg.norm(relative[valid], axis=-1)).mean(axis=1)
        else:
            selected_R, selected_r = R[valid], relative[valid]
            q = np.stack((2*selected_R/3, -selected_R/3-selected_r/2,
                          -selected_R/3+selected_r/2), axis=-2)
            energy[valid] = _energies(provider, q).mean(axis=1)
        delta[valid] = energy[valid]-vref[valid]
    return {"shape": shape, "vref": vref, "energy": energy, "delta": delta,
            "minimum_pair": minimum, "valid": valid}


def log_density(values, lambdas, beta):
    result = np.full(len(lambdas), -np.inf)
    valid = values["valid"]
    result[valid] = (values["shape"][valid]
                     - beta*(values["vref"][valid]+lambdas[valid]*values["delta"][valid]))
    return result


def estimate_reference_domain(reference, *, beads, mass, samples, seed, batch=256):
    """Independent normalized-prior integral of the common bead domain."""
    if beads == 1:
        log_probability = reference.log_classical-reference.log_reference
        probability = float(np.exp(log_probability))
        return {"method": "analytic one-bead classical/reference ratio", "draws": 0,
                "log_probability": log_probability, "probability": probability,
                "binomial_CI95": [probability, probability], "standard_error": 0.,
                "seed": int(seed), "independent_from_chain": True}
    if samples < 1 or batch < 1:
        raise ValueError("positive domain sample and batch counts required")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 8342]))
    accepted = 0
    for start in range(0, samples, batch):
        count = min(batch, samples-start)
        radius, ratio = reference.sample(rng, count)
        eta_r = free_ring(rng, (count, beads, 3), reference.beta, mass/2)
        eta_R = (free_ring(rng, eta_r.shape, reference.beta, 2*mass/3)
                 if reference.surface == "ts" else np.zeros_like(eta_r))
        minimum = path_coordinates(reference.surface, radius, ratio, eta_R, eta_r)[2]
        accepted += int(np.count_nonzero(minimum >= reference.minimum))
    if not accepted:
        raise ValueError("independent reference-domain sampling had no accepted paths; normalization unresolved")
    probability = accepted/samples
    interval = [0. if not accepted else float(beta_distribution.ppf(.025, accepted, samples-accepted+1)),
                1. if accepted == samples else float(beta_distribution.ppf(.975, accepted+1, samples-accepted))]
    return {"method": "independent quadrature-centroid and exact normalized free-ring draws",
            "draws": samples, "accepted": accepted, "probability": probability,
            "log_probability": float(np.log(probability)),
            "standard_error": float(np.sqrt(probability*(1-probability)/samples)),
            "binomial_CI95": interval, "seed": int(seed), "independent_from_chain": True,
            "uncertainty_note": "Binomial interval only; reference quadrature is a separate deterministic convergence check."}


def importance_diagnostic(provider, reference, *, beads, mass, samples=8192,
                          seed=319037, batch=128):
    """Excluded pilot: independent reference importance weights and tail diagnostics.

    This estimates Z_Q/Z_classical directly, including rejected bead-domain
    draws as zero weight. Its ESS and largest normalized weight diagnose whether
    direct perturbation is plausible; neither a point estimate nor a large
    empirical ESS proves that an unobserved rare tail has been sampled.
    """
    if min(beads, mass, samples, batch) <= 0:
        raise ValueError("positive importance diagnostic parameters required")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 8343]))
    observations = np.empty((samples, 4))
    log_weights = np.full(samples, -np.inf)
    accepted = 0
    for start in range(0, samples, batch):
        count = min(batch, samples-start)
        radius, ratio = reference.sample(rng, count)
        eta_r = free_ring(rng, (count, beads, 3), reference.beta, mass/2)
        eta_R = (free_ring(rng, eta_r.shape, reference.beta, 2*mass/3)
                 if reference.surface == "ts" else np.zeros_like(eta_r))
        values = evaluate_paths(provider, reference, radius, ratio, eta_R, eta_r)
        part = log_weights[start:start+count]
        part[values["valid"]] = -reference.beta*values["delta"][values["valid"]]
        observations[start:start+count] = np.column_stack((radius, ratio,
            values["minimum_pair"], np.where(values["valid"], values["delta"], np.nan)))
        accepted += int(values["valid"].sum())
    if not accepted:
        raise ValueError("importance pilot had no physical-domain paths")
    total = float(logsumexp(log_weights))
    normalized = np.exp(log_weights-total)
    effective = float(1/np.sum(normalized**2))
    offset = reference.log_reference-reference.log_classical
    estimate = offset+total-np.log(samples)
    half = samples//2
    halves = [offset+float(logsumexp(part))-np.log(len(part))
              for part in (log_weights[:half], log_weights[half:])] if half else []
    report = {"schema": "pyrpmd.h3.surface-importance-pilot.v1", "phase": "excluded calibration",
              "surface": reference.surface, "beads": beads, "seed": seed,
              "samples": samples, "accepted_domain": accepted,
              "log_quantum_classical_ratio": float(estimate),
              "quantum_classical_ratio": _finite_exp(estimate),
              "effective_importance_samples": effective,
              "effective_fraction": effective/samples,
              "largest_normalized_weight": float(normalized.max()),
              "top_one_percent_weight_fraction": float(np.sort(normalized)[-max(1, (samples+99)//100):].sum()),
              "first_second_half_log_ratios": [value if np.isfinite(value) else None for value in halves],
              "observation_columns": ["centroid_radius_bohr", "angle_ratio", "minimum_bead_pair_bohr",
                                      "delta_energy_hartree_nan_outside_domain"],
               "literature_reproduced": False,
              "limitations": ["Excluded from primary inference; used only for overlap and tail assessment.",
                  "Empirical importance ESS cannot exclude unsampled rare high-weight paths.",
                  "Reference centroid quadrature requires an independent refinement check."]}
    return report, {"observations": observations, "log_weights": log_weights}


def bennett(delta_from_left, delta_from_right):
    """Return f_right-f_left from d=u_right-u_left evaluated in both states."""
    left, right = np.asarray(delta_from_left, float), np.asarray(delta_from_right, float)
    if (left.ndim != 1 or right.ndim != 1 or not len(left) or not len(right)
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError("BAR needs nonempty finite reduced-energy differences")
    log_counts = np.log(len(left)/len(right))

    def residual(free):
        return (expit(free-left-log_counts).sum()
                - expit(right-free+log_counts).sum())

    low = min(left.min(), right.min())-abs(log_counts)-100.
    high = max(left.max(), right.max())+abs(log_counts)+100.
    free = float(brentq(residual, low, high, xtol=1e-12, rtol=2e-14))
    weights_left = expit(free-left-log_counts)
    weights_right = expit(right-free+log_counts)
    return {"delta_f": free, "samples_left": len(left), "samples_right": len(right),
            "overlap_logistic": float(2*(weights_left.sum()+weights_right.sum())/(len(left)+len(right))),
            "equation_residual": float(residual(free)),
            "uncertainty": "Independent-chain uncertainty required; correlated sweeps are not independent observations."}


def analyze_samples(delta_energy, lambdas, beta, log_reference_over_classical, domain):
    samples, lambdas = np.asarray(delta_energy, float), np.asarray(lambdas, float)
    if samples.ndim != 2 or samples.shape[1] != len(lambdas) or len(samples) < 2:
        raise ValueError("at least two sampled energy differences per lambda required")
    edges = []
    for index, width in enumerate(np.diff(lambdas)):
        row = bennett(beta*width*samples[:, index], beta*width*samples[:, index+1])
        row.update(lambda_left=float(lambdas[index]), lambda_right=float(lambdas[index+1]))
        edges.append(row)
    offset = log_reference_over_classical+domain["log_probability"]
    log_ratio = offset-sum(row["delta_f"] for row in edges)
    mean = samples.mean(axis=0)
    trap = float(trapezoid(mean, lambdas))
    simpson_integral = float(simpson(mean, x=lambdas))
    half = len(samples)//2

    def half_ratio(part):
        return offset-sum(bennett(beta*width*part[:, index], beta*width*part[:, index+1])["delta_f"]
                          for index, width in enumerate(np.diff(lambdas)))

    return {"log_quantum_classical_ratio": float(log_ratio),
            "quantum_classical_ratio": _finite_exp(log_ratio),
            "normalization_log_reference_over_classical": float(log_reference_over_classical),
            "normalization_log_domain_probability": domain["log_probability"],
            "bar_edges": edges, "minimum_BAR_overlap": min(row["overlap_logistic"] for row in edges),
            "mean_delta_energy_hartree": mean,
            "TI_trapezoid_log_ratio": offset-beta*trap,
            "TI_simpson_log_ratio": offset-beta*simpson_integral,
            "first_half_log_ratio": half_ratio(samples[:half]),
            "second_half_log_ratio": half_ratio(samples[half:]),
            "diagnostic_energy_effective_samples": [float(len(samples)/statistical_inefficiency(column))
                                                      if np.ptp(column) else None for column in samples.T],
            "uncertainty_unit": "Independent seed chain; halves and autocorrelation estimates are diagnostics only.",
             "literature_reproduced": False}


def harmonic_log_ratio(beta, omega, beads, dimensions=1):
    """Exact primitive finite-P harmonic partition divided by its classical limit."""
    if min(beta, omega, beads, dimensions) <= 0:
        raise ValueError("positive harmonic parameters required")
    argument = beads*np.arcsinh(beta*omega/(2*beads))
    return float(dimensions*(np.log(beta*omega)-argument-np.log(-np.expm1(-2*argument))))


def _provenance(provider):
    names = ("benchmarks/h3/surface_partition.py", "benchmarks/h3/mixing_sampling.py",
             "benchmarks/h3/pes.py", "benchmarks/h3/classical_tst.py", "src/pyrpmd/constants.py",
             "src/pyrpmd/analysis/fes.py", "src/pyrpmd/checkpoint.py", "src/pyrpmd/io.py")
    model = (provider.metadata if isinstance(provider, BKMP2Potential)
             else getattr(provider, "partition_identity", None))
    if model is None:
        raise ValueError("an injected benchmark potential needs an explicit partition_identity")
    return {"source_sha256": {name: _sha(ROOT/name) for name in names}, "model": model,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform()}}


def run(output, library=None, *, potential=None, surface="ts", beads=128, seed=319031,
        warmup=1000, production=5000, lambdas=None, temperature=300., mass=MASS_AU,
        minimum=.5, maximum=8., radial_order=192, angular_order=96,
        domain_samples=16384, radius_step=.035, ratio_step=.03, alpha_R=.35, alpha_r=.30,
        swap_interval=5, checkpoint_interval=1000, sample_interval=1,
        diagnostic_interval=10, snapshot_interval=0, phase="calibration", resume=False, stop_after=None):
    """Run one fixed-protocol replica-exchange chain; never stop on target agreement."""
    integer_options = {"beads": beads, "seed": seed, "warmup": warmup, "production": production,
                       "radial_order": radial_order, "angular_order": angular_order,
                       "domain_samples": domain_samples, "swap_interval": swap_interval,
                       "checkpoint_interval": checkpoint_interval, "sample_interval": sample_interval,
                       "diagnostic_interval": diagnostic_interval, "snapshot_interval": snapshot_interval}
    for name, value in integer_options.items():
        lower = 0 if name in ("warmup", "seed", "snapshot_interval") else 1
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < lower:
            raise ValueError(f"invalid integer {name}")
    if stop_after is not None and (isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after < 1):
        raise ValueError("stop_after must be a positive integer or omitted")
    if (surface not in ("ts", "reactant") or phase not in ("calibration", "production")
            or min(temperature, mass, radius_step, ratio_step) <= 0
            or not np.isfinite([temperature, mass, minimum, maximum, radius_step, ratio_step, alpha_R, alpha_r]).all()
            or not 0 <= minimum < maximum < 20 or not 0 < alpha_R <= 1 or not 0 < alpha_r <= 1
            or min(radial_order, angular_order) < 8 or production < 2*sample_interval):
        raise ValueError("invalid partition sampler parameters")
    if snapshot_interval and surface != "ts":
        raise ValueError("full-q endpoint parent snapshots require the TS surface")
    lambdas = np.linspace(0., 1., 25)**2 if lambdas is None else np.asarray(lambdas, float)
    if (lambdas.ndim != 1 or len(lambdas) < 2 or not np.isfinite(lambdas).all()
            or lambdas[0] != 0 or lambdas[-1] != 1 or np.any(np.diff(lambdas) <= 0)):
        raise ValueError("strictly increasing lambda grid must include exactly 0 and 1")
    if library is not None and potential is not None:
        raise ValueError("provide a library or an injected potential, not both")
    owned = potential is None
    if owned and not .4 <= minimum <= .7:
        raise ValueError("BKMP2 bead cutoff must stay in the explicitly audited .4 to .7 bohr range")
    provider = BKMP2Potential(library) if owned else potential
    beta = 1/(KB_AU*temperature)
    identity = _provenance(provider)
    settings = jsonable(dict(schema="pyrpmd.h3.surface-partition.v1", phase=phase, surface=surface, beads=beads, seed=seed,
                    warmup=warmup, production=production, lambdas=lambdas.tolist(),
                    temperature_K=temperature, beta=beta, mass_au=mass, minimum_bead_pair_bohr=minimum,
                    maximum_centroid_bond_bohr=maximum, radial_order=radial_order, angular_order=angular_order,
                    domain_samples=domain_samples, radius_step=radius_step, ratio_step=ratio_step,
                    alpha_R=alpha_R, alpha_r=alpha_r, swap_interval=swap_interval,
                    checkpoint_interval=checkpoint_interval, sample_interval=sample_interval,
                    diagnostic_interval=diagnostic_interval, snapshot_interval=snapshot_interval, **identity,
                    endpoint_snapshot_source="static equilibrium Monte Carlo at lambda=1; normalized total COM analytically marginalized",
                    endpoint_snapshot_channel="r01=r12<=r02; equivalent channel and Maxwell momenta assigned only by separate child RNG",
                    target="physical bead domain and continuous centroid shape; normalized Jacobi free-ring prior",
                    reference="centroid PES at max(r, minimum); full centroid support (0, maximum)",
                    sampling_is_real_time_dynamics=False, uncertainty_unit="independent seed chain"))
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("use a fresh output directory, or resume identical settings")
    output.mkdir(parents=True, exist_ok=True)
    settings_path = output/"settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("partition settings/source identity changed; use a new run")
    write_json(settings_path, settings)
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    reference = reference_quadrature(provider, surface, beta, minimum=minimum, maximum=maximum,
                                      radial_order=radial_order, angular_order=angular_order)
    write_json(output/"reference.json", reference.report())
    domain_path = output/"reference-domain.json"
    if resume and domain_path.exists():
        domain = json.loads(domain_path.read_text())
    else:
        domain = estimate_reference_domain(reference, beads=beads, mass=mass, samples=domain_samples, seed=seed)
        write_json(domain_path, domain)
    domain_hash = _sha(domain_path)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 8341]))
    n = len(lambdas)
    sample_capacity = (production+sample_interval-1)//sample_interval
    diagnostic_capacity = (production+diagnostic_interval-1)//diagnostic_interval
    samples = np.empty((sample_capacity, n))
    diagnostics = np.empty((diagnostic_capacity, n, 8))
    snapshot_capacity = production//snapshot_interval if snapshot_interval else 0
    snapshots = np.empty((snapshot_capacity, beads, 3, 3))
    snapshot_sweeps = (warmup+np.arange(1, snapshot_capacity+1)*snapshot_interval)
    accepted = np.zeros((4, n), dtype=np.int64); proposed = np.zeros_like(accepted)
    swap_attempts = np.zeros(n-1, dtype=np.int64); swap_accepts = np.zeros_like(swap_attempts)
    lineage = np.arange(n); last_extreme = np.full(n, -1, dtype=np.int64)
    last_extreme[0], last_extreme[-1] = 0, 1
    endpoint_traversals = np.zeros((2, n), dtype=np.int64)
    checkpoint = output/"checkpoint.npz"
    completed = 0
    if resume and checkpoint.exists():
        with load_checkpoint(checkpoint, require_integrity=True) as old:
            if str(old["fingerprint"]) != fingerprint or str(old["domain_sha256"]) != domain_hash:
                raise ValueError("partition checkpoint identity/domain normalization differs")
            radius, ratio = old["radius"].copy(), old["ratio"].copy()
            eta_R, eta_r = old["eta_R"].copy(), old["eta_r"].copy()
            completed = int(old["completed"])
            if not 0 <= completed <= warmup+production:
                raise ValueError("invalid partition checkpoint progress")
            count = max(0, completed-warmup)
            ns = (count+sample_interval-1)//sample_interval
            nd = (count+diagnostic_interval-1)//diagnostic_interval
            nq = count//snapshot_interval if snapshot_interval else 0
            if old["delta_energy"].shape != (ns, n) or old["diagnostics"].shape != (nd, n, 8):
                raise ValueError("partition checkpoint observation count differs")
            if (old["endpoint_q"].shape != (nq, beads, 3, 3)
                    or not np.array_equal(old["endpoint_completed_sweeps"], snapshot_sweeps[:nq])):
                raise ValueError("partition checkpoint endpoint snapshot count/order differs")
            samples[:ns], diagnostics[:nd] = old["delta_energy"], old["diagnostics"]
            snapshots[:nq] = old["endpoint_q"]
            accepted, proposed = old["accepted"].copy(), old["proposed"].copy()
            swap_attempts, swap_accepts = old["swap_attempts"].copy(), old["swap_accepts"].copy()
            lineage, last_extreme = old["lineage"].copy(), old["last_extreme"].copy()
            endpoint_traversals = old["endpoint_traversals"].copy()
            rng.bit_generator.state = json.loads(str(old["rng"]))
    else:
        radius, ratio = reference.sample(rng, n)
        eta_R = np.zeros((n, beads, 3)); eta_r = np.zeros_like(eta_R)
        # Initial centroids are classical-domain draws. The low-r reference
        # plateau remains accessible to subsequent continuous shape proposals.
        outside = radius < minimum
        while outside.any():
            radius[outside], ratio[outside] = reference.sample(rng, int(outside.sum()))
            outside = radius < minimum
    values = evaluate_paths(provider, reference, radius, ratio, eta_R, eta_r)
    logprob = log_density(values, lambdas, beta)
    if not np.isfinite(logprob).all():
        raise ValueError("invalid initial or resumed partition state")
    started = time.perf_counter()

    def save():
        if _provenance(provider) != identity:
            raise RuntimeError("partition sources changed; state not committed")
        count = max(0, completed-warmup)
        ns = (count+sample_interval-1)//sample_interval
        nd = (count+diagnostic_interval-1)//diagnostic_interval
        nq = count//snapshot_interval if snapshot_interval else 0
        atomic_savez(checkpoint, kind="h3.surface-partition", fingerprint=fingerprint,
                     domain_sha256=domain_hash, radius=radius, ratio=ratio, eta_R=eta_R, eta_r=eta_r,
                     completed=completed, delta_energy=samples[:ns], diagnostics=diagnostics[:nd],
                     endpoint_q=snapshots[:nq], endpoint_completed_sweeps=snapshot_sweeps[:nq],
                     accepted=accepted, proposed=proposed, swap_attempts=swap_attempts,
                     swap_accepts=swap_accepts, lineage=lineage, last_extreme=last_extreme,
                     endpoint_traversals=endpoint_traversals, rng=json.dumps(rng.bit_generator.state))
        write_json(output/"progress.json", {"completed_sweeps": completed, "production_sweeps": count,
                   "elapsed_this_call_seconds": time.perf_counter()-started,
                   "acceptance_per_move": (accepted/np.maximum(proposed, 1)).mean(axis=1),
                    "literature_reproduced": False})

    try:
        if not checkpoint.exists():
            save()
        limit = warmup+production if stop_after is None else min(warmup+production, completed+stop_after)
        while completed < limit:
            for move in range(4):
                if (move == 1 and surface == "reactant") or (move >= 2 and beads == 1) or (move == 2 and surface == "reactant"):
                    continue
                trial_radius, trial_ratio, trial_R, trial_r = radius, ratio, eta_R, eta_r
                if move == 0:
                    trial_radius = radius+rng.normal(scale=radius_step, size=n)
                elif move == 1:
                    trial_ratio = ratio+rng.normal(scale=ratio_step, size=n)
                else:
                    path, alpha, reduced_mass = ((eta_R, alpha_R, 2*mass/3) if move == 2 else (eta_r, alpha_r, mass/2))
                    trial = np.sqrt(1-alpha*alpha)*path+alpha*free_ring(rng, path.shape, beta, reduced_mass)
                    trial -= trial.mean(axis=1, keepdims=True)
                    if move == 2:
                        trial_R = trial
                    else:
                        trial_r = trial
                candidate = evaluate_paths(provider, reference, trial_radius, trial_ratio, trial_R, trial_r)
                proposed_logprob = log_density(candidate, lambdas, beta)
                take = np.log(rng.random(n)) < proposed_logprob-logprob
                proposed[move] += 1; accepted[move] += take
                for current, new in ((radius, trial_radius), (ratio, trial_ratio), (eta_R, trial_R), (eta_r, trial_r)):
                    if new is not current:
                        current[take] = new[take]
                for name in values:
                    values[name][take] = candidate[name][take]
                logprob[take] = proposed_logprob[take]
            if (completed+1) % swap_interval == 0:
                left = np.arange(int(rng.integers(2)), n-1, 2)
                right = left+1
                # log(new/old)=(lambda_right-lambda_left)*beta*(delta_right-delta_left).
                logratio = beta*(lambdas[right]-lambdas[left])*(values["delta"][right]-values["delta"][left])
                take = np.log(rng.random(len(left))) < logratio
                swap_attempts[left] += 1; swap_accepts[left] += take
                a, b = left[take], right[take]
                for array in (radius, ratio, eta_R, eta_r, lineage, *values.values()):
                    array[a], array[b] = array[b].copy(), array[a].copy()
                logprob = log_density(values, lambdas, beta)
            for endpoint, walker in ((0, lineage[0]), (1, lineage[-1])):
                if last_extreme[walker] == 1-endpoint:
                    endpoint_traversals[endpoint, walker] += 1
                last_extreme[walker] = endpoint
            completed += 1
            if completed > warmup:
                index = completed-warmup-1
                if index % sample_interval == 0:
                    samples[index//sample_interval] = values["delta"]
                if index % diagnostic_interval == 0:
                    diagnostics[index//diagnostic_interval] = np.column_stack((radius, ratio, values["vref"],
                        values["energy"], values["minimum_pair"], np.mean(eta_R**2, axis=(1, 2)),
                        np.mean(eta_r**2, axis=(1, 2)), lineage))
                if snapshot_interval and (index+1) % snapshot_interval == 0:
                    # Deterministic observation only: no additional RNG draws.
                    R, relative, _ = path_coordinates(surface, radius[-1:], ratio[-1:],
                                                       eta_R[-1:], eta_r[-1:])
                    snapshots[(index+1)//snapshot_interval-1] = np.stack((
                        2*R[0]/3, -R[0]/3-relative[0]/2, -R[0]/3+relative[0]/2), axis=-2)
            if completed % checkpoint_interval == 0:
                save()
        save()
    finally:
        if owned:
            provider.close()
    count = max(0, completed-warmup)
    ns = (count+sample_interval-1)//sample_interval
    analysis = (analyze_samples(samples[:ns], lambdas, beta, reference.log_reference-reference.log_classical, domain)
                if ns >= 2 else None)
    report = {"schema": settings["schema"], "phase": phase, "surface": surface, "seed": seed, "beads": beads,
              "complete": completed == warmup+production, "completed_sweeps": completed,
              "production_samples_per_lambda": ns, 
              "endpoint_snapshot_count": count//snapshot_interval if snapshot_interval else 0,
              "endpoint_snapshot_interval": snapshot_interval,
              "endpoint_snapshot_source": settings["endpoint_snapshot_source"],
              "endpoint_snapshot_channel": settings["endpoint_snapshot_channel"],
              "literature_reproduced": False, "elapsed_this_call_seconds": time.perf_counter()-started,
              "reference": reference.report(), "reference_domain": domain, "analysis": analysis,
              "move_names": ["centroid_radius", "centroid_angle_ratio", "R_pcn", "r_pcn"],
              "acceptance_per_lambda": accepted/np.maximum(proposed, 1),
              "swap_attempts": swap_attempts, "swap_accepts": swap_accepts,
              "endpoint_traversals_per_walker": endpoint_traversals,
              "diagnostic_columns": ["centroid_radius_bohr", "angle_ratio", "reference_energy_hartree",
                  "mean_bead_energy_hartree", "minimum_bead_pair_bohr", "R_mean_coordinate_variance_bohr2",
                  "r_mean_coordinate_variance_bohr2", "walker_lineage"],
              "checkpoint_sha256": _sha(checkpoint), "source_unchanged": _provenance(provider) == identity,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "limitations": ["Finite-P primitive equilibrium partition, not a recrossing coefficient or a full RPMD rate.",
                              "The reactant is the asymptotic isolated diatom with the declared 99-bohr spectator convention.",
                              "Centroid quadrature, bead-domain probability, independent-seed sampling and bead-number convergence require separate checks."]}
    write_json(output/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--surface", choices=("ts", "reactant"), required=True)
    parser.add_argument("--beads", type=int, default=128)
    parser.add_argument("--seed", type=int, default=319031)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--production", type=int, default=5000)
    parser.add_argument("--domain-samples", type=int, default=16384)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--snapshot-interval", type=int, default=0)
    parser.add_argument("--phase", choices=("calibration", "production"), default="calibration")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    report = run(**args)
    print(json.dumps({"complete": report["complete"], "beads": report["beads"], "surface": report["surface"],
                      "analysis": report["analysis"]}, default=lambda value: value.tolist(), indent=2))


if __name__ == "__main__":
    main()
