"""Recalculate the excluded HCH4 pilot from saved values only.

No producer, PyRPMD, potential library or random generator is imported or used.
This is an independent-calculation replay by the producer's author, not an
independent-author review. Its numerical tolerances are reconstruction checks,
not modified physical or proposal-efficiency acceptance criteria.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp, stdtrit

SEEDS = tuple(range(2026112201, 2026112209))
CHANNELS = (0, 2, 3, 4)
TARGETS = ("alpha_0.999", "alpha_1.007", "entry")
DOMAINS = {"inner": (1.6, 4., .9), "original": (1.4, 4.5, .7), "outer": (1.2, 5., .6)}
TOLERANCES = {
    "geometry": {"atol": 2e-11, "rtol": 2e-12},
    "density": {"atol": 2e-10, "rtol": 2e-12},
    "calibration": {"atol": 2e-7, "rtol": 2e-9},
    "analysis": {"atol": 5e-12, "rtol": 2e-10},
}


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def identify(path):
    path = Path(path)
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class Comparison:
    def __init__(self):
        self.checked = Counter()
        self.maximum_absolute_error = Counter()

    def check(self, actual, expected, category, label):
        if actual is None or expected is None or isinstance(expected, (str, bool)):
            require(actual == expected, f"{label}: exact value differs")
            self.checked[category] += 1
            return
        a, b = np.asarray(actual), np.asarray(expected)
        require(a.shape == b.shape, f"{label}: shape differs")
        require(np.isfinite(a).all() and np.isfinite(b).all(), f"{label}: nonfinite comparison")
        if a.dtype.kind in "biu" and b.dtype.kind in "biu":
            require(np.array_equal(a, b), f"{label}: integer value differs")
        else:
            delta = float(np.max(np.abs(a-b), initial=0.))
            self.maximum_absolute_error[category] = max(self.maximum_absolute_error[category], delta)
            require(np.allclose(a, b, **TOLERANCES[category]), f"{label}: reconstruction error {delta}")
        self.checked[category] += a.size

    def tree(self, actual, expected, label="analysis"):
        if isinstance(expected, dict):
            require(set(actual) == set(expected), f"{label}: fields differ")
            for key in expected:
                self.tree(actual[key], expected[key], label+"/"+str(key))
        elif isinstance(expected, list):
            require(len(actual) == len(expected), f"{label}: length differs")
            for index, value in enumerate(expected):
                self.tree(actual[index], value, label+"/"+str(index))
        else:
            self.check(actual, expected, "analysis", label)


def stereo(t):
    t = np.asarray(t, float)
    d = 1.+np.dot(t, t)
    return np.array([2*t[0]/d, 2*t[1]/d, (2.-d)/d]), 4./d**2


def make_methane(z, h):
    z = np.asarray(z, float)
    if z[0] <= 0 or z[1] <= 0 or abs(z[2]) >= 1:
        return None
    others = sorted(set(CHANNELS)-{h})
    q = np.zeros((6, 3))
    q[h, 2] = z[0]
    q[others[0]] = [z[1]*math.sqrt(1-z[2]**2), 0., z[1]*z[2]]
    q[others[1:]] = z[3:].reshape(2, 3)
    return q


def coordinate_measure(z, u, h, alpha, constants, length=30.):
    """Use explicit 18D Cartesian gradients, independently of compact metric."""
    mass = np.asarray(constants["masses_electron_mass"])
    beta = constants["beta_hartree_inverse"]
    c = constants["reference_difference_bohr"]
    z, u = np.asarray(z, float), np.asarray(u, float)
    require(np.isfinite(z).all() and np.isfinite(u).all(), "nonfinite latent coordinate")
    require(abs(np.dot(u, u)-1.) < 2e-12, "nonunit incident direction")
    answer = {"coordinates_bohr": None, "common_zero_reason": None,
              "domains": dict.fromkeys(DOMAINS, False), "log_geometric_measure": None,
              "minimum_pair_bohr": None, "CH_radii_bohr": None}
    q = make_methane(z, h)
    if q is None:
        answer["common_zero_reason"] = "invalid_chart"
        return answer
    bonds = np.linalg.norm(q[list(CHANNELS)], axis=1)
    answer["CH_radii_bohr"] = bonds.tolist()
    if np.any((bonds < 1.2) | (bonds > 5.)):
        answer["common_zero_reason"] = "outside_outer_methane_domain"
        return answer
    methane_mass = math.fsum(mass[:5])
    center = (mass[:5, None]*q[:5]).sum(axis=0)/methane_mass
    if alpha is None:
        q[5] = center+length*u
        direction = (q[5]-center)/np.linalg.norm(q[5]-center)
        gradient = np.zeros((6, 3))
        gradient[:5] = mass[:5, None]/methane_mass*direction
        gradient[5] = -direction
        metric = np.sum(gradient**2/mass[:, None])
        measure = 2*math.log(z[0])+2*math.log(z[1])+2*math.log(length)+.5*math.log(metric/(2*math.pi*beta))
    else:
        vec = q[h]-center
        def surface(radius):
            return (1.-alpha)*(length-np.linalg.norm(vec+radius*u))+alpha*(c+z[0]-radius)
        at_zero = surface(0.)
        if at_zero <= 0:
            answer["common_zero_reason"] = "no_positive_root"
            return answer
        upper = at_zero/(alpha-abs(1.-alpha))+1e-12
        radius = brentq(surface, 0., upper, xtol=5e-15, rtol=4*np.finfo(float).eps)
        require(abs(surface(radius)) < 1e-11, "independent radial root residual")
        q[5] = q[h]+radius*u
        direction = (q[5]-center)/np.linalg.norm(q[5]-center)
        grad0 = np.zeros((6, 3)); grad0[:5] = mass[:5, None]/methane_mass*direction; grad0[5] = -direction
        grad1 = np.zeros((6, 3)); bhat = q[h]/np.linalg.norm(q[h])
        incident = (q[5]-q[h])/np.linalg.norm(q[5]-q[h])
        grad1[h] = bhat+incident; grad1[1] = -bhat; grad1[5] = -incident
        gradient = (1-alpha)*grad0+alpha*grad1
        metric = np.sum(gradient**2/mass[:, None])
        derivative = float(gradient[5]@u)
        require(metric > 0 and derivative < 0, "nonpositive flux measure or radial derivative")
        measure = (2*math.log(z[0])+2*math.log(z[1])+2*math.log(radius)
                   -math.log(-derivative)+.5*math.log(metric/(2*math.pi*beta)))
    answer["coordinates_bohr"] = q.tolist()
    distances = [float(np.linalg.norm(q[a]-q[b])) for a in range(6) for b in range(a)]
    minimum = min(distances); answer["minimum_pair_bohr"] = minimum
    channel_values = np.array([c+np.linalg.norm(q[k])-np.linalg.norm(q[5]-q[k]) for k in CHANNELS])
    active = CHANNELS[int(np.argmax(channel_values))]
    R = np.linalg.norm(q[5]-center)
    if minimum < .6:
        answer["common_zero_reason"] = "outside_outer_pair_domain"
    elif alpha is not None and active != h:
        answer["common_zero_reason"] = "inactive_max_channel"
    elif length-R-float(channel_values.max()) <= 0:
        answer["common_zero_reason"] = "disconnected_xi_component"
    else:
        answer["log_geometric_measure"] = measure
        answer["domains"] = {key: bool(np.all((bonds >= lo)&(bonds <= hi)) and minimum >= pair)
                             for key, (lo, hi, pair) in DOMAINS.items()}
    return answer


class FullMixture:
    """Precomputed precision matrices and direct joint-density balance weights."""
    def __init__(self, models, entry=False):
        self.entry = entry
        self.parts = []
        for target in (("entry",) if entry else TARGETS[:2]):
            m = models[target]
            signs = np.ones(len(m["mean_x"])); signs[[4, 7]] = -1
            if not entry:
                signs[-1] = -1
            for sign in (np.ones(len(signs)), signs):
                mean = np.asarray(m["mean_x"])*sign
                covariance = np.asarray(m["covariance_x"])*np.outer(sign, sign)
                for student, fraction in ((False, .75/(2 if entry else 4)), (True, .125/(2 if entry else 4))):
                    matrix = covariance*(2.4 if student else 1.)
                    positive, logdet = np.linalg.slogdet(matrix)
                    require(positive == 1, "nonpositive proposal covariance")
                    self.parts.append((mean, np.linalg.inv(matrix), logdet, student, math.log(fraction)))

    def __call__(self, z, u):
        z, u = np.asarray(z), np.asarray(u)
        low = np.array([1.2, 1.2, -1, *([-5]*6)])
        high = np.array([5., 5., 1, *([5]*6)])
        terms = []
        if np.all((z >= low)&(z <= high)):
            terms.append(-math.log(8*np.prod(high-low)*4*math.pi))
        if self.entry:
            x, jac = z, 4*math.pi
        elif u[2] == -1:
            return float(logsumexp(terms)) if terms else -math.inf
        else:
            x = np.concatenate((z, u[:2]/(1+u[2])))
            jac = (1+u[2])**2
        for mean, precision, logdet, student, logfraction in self.parts:
            displacement = x-mean
            radial = float(displacement@precision@displacement)
            d = len(x)
            if student:
                density = (math.lgamma((d+5)/2)-math.lgamma(2.5)-.5*(d*math.log(5*math.pi)+logdet)
                           -(d+5)/2*math.log1p(radial/5))
            else:
                density = -.5*(d*math.log(2*math.pi)+logdet+radial)
            terms.append(logfraction+density-math.log(jac))
        return float(logsumexp(terms))


def average_log(values):
    x = np.asarray([v if v is not None else -np.inf for v in values])
    return None if not np.isfinite(x).any() else float(logsumexp(x)-math.log(len(x)))


def summary(values):
    positive = np.array([v for v in values if v is not None])
    if not len(positive):
        return {"draws": len(values), "positive_weights": 0, "log_mean": None, "empirical_ESS": 0.,
                "maximum_normalized_weight": None, "largest_one_percent_weight_mass": None}
    scaled = np.exp(positive-positive.max()); total = math.fsum(scaled)
    probabilities = scaled/total
    return {"draws": len(values), "positive_weights": len(positive),
            "log_mean": float(positive.max()+math.log(total/len(values))),
            "empirical_ESS": float(1./math.fsum(probabilities**2)),
            "maximum_normalized_weight": float(max(probabilities)),
            "largest_one_percent_weight_mass": float(math.fsum(sorted(probabilities, reverse=True)[:math.ceil(.01*len(values))]))}


def integral_vector(rows, weights):
    output = {}
    for target in TARGETS:
        selected = [weights[row["index"]]["original"] for row in rows if row["target"] == target]
        mean = average_log(selected)
        # Four equal-size channel strata: sum_h sum_i(w_hi)/n_h = 4*mean(w).
        output[target] = None if mean is None else mean+(0. if target == "entry" else math.log(4.))
    return output


def covariance_ratio(first, second):
    require(len(first) == len(second) == 8, "eight complete seed vectors required")
    a = np.asarray([-np.inf if x is None else x for x in first])
    b = np.asarray([-np.inf if x is None else x for x in second])
    if not np.isfinite(a).any() or not np.isfinite(b).any():
        return {"log_ratio_of_pooled_means": None, "resolved": False,
                "reason": "zero pooled integral; no epsilon substitution", "seed_vectors": 8}
    scale_a, scale_b = float(max(a)), float(max(b))
    A, B = np.exp(a-scale_a), np.exp(b-scale_b)
    mean_a, mean_b = math.fsum(A)/8, math.fsum(B)/8
    estimate = scale_a-scale_b+math.log(mean_a)-math.log(mean_b)
    influence = A/mean_a-B/mean_b
    centered = influence-math.fsum(influence)/8
    se = math.sqrt(math.fsum(centered**2)/56)
    half = float(stdtrit(7, .975))*se
    return {"log_ratio_of_pooled_means": estimate, "resolved": True, "seed_vectors": 8,
            "degrees_of_freedom": 7, "seed_influence": influence.tolist(),
            "nominal_delta_standard_error": se, "nominal_delta_t95_interval": [estimate-half, estimate+half],
            "interpretation": "Conditional small-sample Monte Carlo approximation; not a tail bound or physical-rate confidence interval."}


def independent_analysis(rows, controls, weights, control_weights):
    seed_values = {seed: integral_vector([row for row in rows if row["seed"] == seed], weights) for seed in SEEDS}
    pooled = integral_vector(rows, weights)
    diagnostics, channels, gates = {}, {}, {}
    for target in TARGETS:
        selected = [row for row in rows if row["target"] == target]
        logs = [weights[r["index"]]["original"] for r in selected]
        stats = summary(logs)
        leave = [abs(average_log([seed_values[s][target] for s in SEEDS if s != omitted])-pooled[target]) for omitted in SEEDS]
        stats["leave_one_seed_out_absolute_log_integral_change"] = leave
        total = logsumexp([x for x in logs if x is not None])
        contribution = {}
        for component in sorted({r["component"] for r in selected}):
            vals = [weights[r["index"]]["original"] for r in selected if r["component"] == component]
            item = summary(vals)
            finite = [v for v in vals if v is not None]
            item["observed_fraction_of_total_weight"] = 0. if not finite else float(math.exp(logsumexp(finite)-total))
            contribution[component] = item
        stats["component_contributions"] = contribution
        stats["nested_domains"] = {d: summary([weights[r["index"]][d] for r in selected]) for d in DOMAINS}
        shell = [weights[r["index"]]["outer"] if not r["domains"]["original"] else None for r in selected]
        stats["outer_shell"] = dict(summary(shell), positive_seed_count=len({r["seed"] for r, v in zip(selected, shell) if v is not None}),
                                    zero_observation_upper_bound_claimed=False)
        diagnostics[target] = stats
        gates[target] = bool(stats["empirical_ESS"] >= 128 and stats["maximum_normalized_weight"] <= .025
                             and stats["largest_one_percent_weight_mass"] <= .20 and max(leave) <= .25)
        if target != "entry":
            for channel in CHANNELS:
                key = f"{target}:channel{channel}"
                channels[key] = summary([weights[r["index"]]["original"] for r in selected if r["channel"] == channel])
                gates[key] = bool(channels[key]["empirical_ESS"] >= 32 and channels[key]["maximum_normalized_weight"] <= .10)
    ratio = {target: covariance_ratio([seed_values[s][target] for s in SEEDS], [seed_values[s]["entry"] for s in SEEDS]) for target in TARGETS[:2]}
    pairs = [[], []]
    for seed in SEEDS:
        selected = [r for r in controls if r["seed"] == seed]
        for i, (collection, distance) in enumerate(((weights, 30.), (control_weights, 60.))):
            values = [collection[r["paired_main_index"] if i == 0 else r["index"]]["original"] for r in selected]
            pairs[i].append(average_log([None if v is None else v-2*math.log(distance) for v in values]))
    separation = covariance_ratio(pairs[1], pairs[0])
    separation.update(draws_at_each_separation=128, matched_draws_per_seed=16,
                      primary_entry_denominator_draws_per_seed=128, full_finite_Rinf_equivalence_claimed=False)
    return {"all_eight_seeds": list(SEEDS), "pooled_log_integrals_without_common_rotation": pooled,
            "seed_log_integrals_without_common_rotation": {str(s): value for s, value in seed_values.items()},
            "dimensionless_flux_ratio_diagnostics": ratio,
            "paired_alpha_log_integral_difference": covariance_ratio([seed_values[s][TARGETS[1]] for s in SEEDS], [seed_values[s][TARGETS[0]] for s in SEEDS]),
            "entry_only_separation_control": separation, "weight_diagnostics": diagnostics,
            "channel_weight_diagnostics": channels, "efficiency_screen_components": gates,
            "original_domain_efficiency_screen_passed": all(gates.values()),
            "efficiency_screen_is_scientific_validation": False, "automatic_continuation": False}


def check_membership(rows, controls):
    require([r["index"] for r in rows] == list(range(3072)), "main inventory/order")
    require([r["index"] for r in controls] == list(range(3072, 3200)), "control inventory/order")
    for seed in SEEDS:
        own = [r for r in rows if r["seed"] == seed]
        require(len(own) == 384, "complete seed membership")
        for channel in CHANNELS:
            selected = [[r for r in own if r["target"] == t and r["channel"] == channel] for t in TARGETS[:2]]
            for pair in zip(*selected):
                for key in ("latent_id", "shape", "direction", "component", "component_draw"):
                    require(pair[0][key] == pair[1][key], "alpha pairing differs")
            expected = {"uniform": 4, **{f"{kind}:{target}:{mirror}": count for kind, count in (("normal", 6), ("student5", 1))
                        for target in TARGETS[:2] for mirror in ("positive", "negative")}}
            for group in selected:
                require(Counter(r["component"] for r in group) == expected, "TS exact stratum allocation")
                for component, count in expected.items():
                    require(sorted(r["component_draw"] for r in group if r["component"] == component) == list(range(count)), "TS draw indices")
        entry = [r for r in own if r["target"] == "entry"]
        expected = {"uniform": 16, **{f"{kind}:entry:{mirror}": count for kind, count in (("normal", 48), ("student5", 8)) for mirror in ("positive", "negative")}}
        require(Counter(r["component"] for r in entry) == expected and all(r["channel"] is None for r in entry), "single entry chart")
        for component, count in expected.items():
            require(sorted(r["component_draw"] for r in entry if r["component"] == component) == list(range(count)), "entry draw indices")
        chosen = [r for r in controls if r["seed"] == seed]
        require(len(chosen) == len({r["paired_main_index"] for r in chosen}) == 16, "paired control unique count")
        expected_control = {k: 2 if k == "uniform" else 6 if k.startswith("normal") else 1 for k in expected}
        require(Counter(r["component"] for r in chosen) == expected_control, "paired control allocation")
        for r in chosen:
            parent = rows[r["paired_main_index"]]
            require(parent["target"] == r["target"] == "entry" and parent["separation_bohr"] == 30. and r["separation_bohr"] == 60., "control route")
            for key in ("seed", "latent_id", "shape", "direction", "component", "component_draw"):
                require(parent[key] == r[key], "same16 entry draws required")
            require(0 <= r["component_draw"] < expected_control[r["component"]], "fixed pre-energy control subset")


def calibration_replay(initial, followup, proposal, energies, constants, offset, compare):
    models, objectives = {}, {}
    for stage, ledger in (("initial_calibration", initial), ("followup_calibration", followup)):
        for row in ledger["rows"]:
            target = row["target"]; x = np.asarray(row["x"])
            u, jac = (np.array([0., 0., 1.]), 1.) if target == "entry" else stereo(x[9:])
            geometry = coordinate_measure(x[:9], u, 3, None if target == "entry" else float(target.split("_")[1]), constants)
            require(geometry["domains"]["original"] and geometry["common_zero_reason"] is None, "all calibration points in original domain")
            for key in geometry:
                if key == "domains":
                    require(geometry[key] == row[key], "calibration domains differ")
                else:
                    compare.check(geometry[key], row[key], "geometry", f"{stage}/{row['index']}/{key}")
            training = geometry["log_geometric_measure"]+math.log(jac)
            compare.check(training, row["log_training_measure"], "geometry", "training measure")
            objectives[(target, row["tag"])] = constants["beta_hartree_inverse"]*(energies[(stage, row["index"])]-offset)-training
    require([r["index"] for r in initial["rows"]] == list(range(711)) and [r["index"] for r in followup["rows"]] == list(range(711, 800)), "calibration inventory")
    for target in TARGETS:
        spec = initial["specifications"][target]; center = np.asarray(spec["x0"]); scale = np.asarray(spec["scales"]); dimension = len(center)
        selected = [r for r in initial["rows"] if r["target"] == target]
        tags = [r["tag"] for r in selected]
        expected = {"center", *[f"{kind}:{i}:{s}" for kind in ("diag", "refine") for i in range(dimension) for s in (-1, 1)],
                    *[f"cross:{i}:{j}:{a}:{b}" for i in range(dimension) for j in range(i+1, dimension) for a in (-1, 1) for b in (-1, 1)]}
        require(len(tags) == len(expected) and set(tags) == expected, "full central Hessian stencil")
        for row in selected:
            compare.check(np.asarray(row["x"]), center+scale*np.asarray(row["y"]), "geometry", "scaled coordinate back-transform")
        get = lambda tag: objectives[(target, tag)]
        f0 = get("center"); step = .02
        H = np.empty((dimension, dimension)); gradient = np.empty(dimension); refined = np.empty(dimension)
        for i in range(dimension):
            gradient[i] = (get(f"diag:{i}:1")-get(f"diag:{i}:-1"))/(2*step)
            H[i, i] = ((get(f"diag:{i}:1")-f0)+(get(f"diag:{i}:-1")-f0))/step**2
            refined[i] = ((get(f"refine:{i}:1")-f0)+(get(f"refine:{i}:-1")-f0))/.01**2
            for j in range(i):
                H[i, j] = H[j, i] = (get(f"cross:{j}:{i}:1:1")+get(f"cross:{j}:{i}:-1:-1")
                                                  -get(f"cross:{j}:{i}:1:-1")-get(f"cross:{j}:{i}:-1:1"))/(4*step**2)
        eigenvalues = np.linalg.eigvalsh(H); delta = np.linalg.solve(H, -gradient)
        covariance = np.diag(scale)@np.linalg.solve(H, np.eye(dimension))@np.diag(scale)
        result = {"H_y": H, "gradient_y": gradient, "delta_y": delta, "mean_x": center+scale*delta,
                  "covariance_x": covariance, "eigenvalues_y": eigenvalues,
                  "Newton_Mahalanobis_length": float(math.sqrt(delta@H@delta)),
                  "maximum_diagonal_refinement_discrepancy": float(np.max(np.abs(H.diagonal()-refined)/np.maximum(1., abs(refined))))}
        require(eigenvalues[0] > 1e-6 and eigenvalues[-1]/eigenvalues[0] <= 1e8 and result["Newton_Mahalanobis_length"] <= 3
                and result["maximum_diagonal_refinement_discrepancy"] <= .05, "unchanged Hessian acceptance checks")
        for key, value in result.items():
            compare.check(value, proposal["models"][target][key], "calibration", target+"/"+key)
            compare.check(value, followup["models"][target][key], "calibration", "followup/"+target+"/"+key)
        saved_axes = np.asarray(proposal["models"][target]["unit_probe_axes_x"])/scale[:, None]
        compare.check(saved_axes.T@H@saved_axes, np.eye(dimension), "calibration", "unit Mahalanobis axes")
        compare.check(H@saved_axes, saved_axes*eigenvalues[None, :], "calibration", "ascending eigenaxes")
        model_rows = [r for r in followup["rows"] if r["target"] == target]
        lookup = {r["tag"]: r for r in model_rows}
        for row in model_rows:
            if row["tag"] == "shifted_center":
                expected_x = result["mean_x"]
            elif row["tag"].startswith("probe:"):
                _, index, sign = row["tag"].split(":")
                expected_x = result["mean_x"]+int(sign)*scale*saved_axes[:, int(index)]
            else:
                signs = np.ones(dimension); signs[[4, 7]] = -1
                if dimension == 11: signs[-1] = -1
                expected_x = np.asarray(lookup[row["reflected_original_tag"]]["x"])*signs
            compare.check(row["x"], expected_x, "calibration", "followup probe location")
        residuals = [abs(objectives[(target, r["tag"])]-objectives[(target, "shifted_center")]-.5) for r in model_rows if r["tag"].startswith("probe:")]
        energy_errors, distance_errors = [], []
        for row in model_rows:
            if "reflected_original_tag" not in row: continue
            original = lookup[row["reflected_original_tag"]]
            energy_errors.append(abs(energies[("followup_calibration", row["index"])]-energies[("followup_calibration", original["index"])]))
            a, b = np.asarray(row["coordinates_bohr"]), np.asarray(original["coordinates_bohr"])
            distance_errors.append(float(np.max(np.abs(np.linalg.norm(a[:, None]-a[None], axis=-1)-np.linalg.norm(b[:, None]-b[None], axis=-1)))))
        require(len(energy_errors) == 8 and max(residuals) <= 5 and max(energy_errors) <= 1e-10 and max(distance_errors) <= 1e-12, "unchanged probe/reflection checks")
        checks = {"maximum_quadratic_probe_residual": max(residuals), "maximum_reflection_energy_difference_hartree": max(energy_errors),
                  "maximum_reflection_distance_difference_bohr": max(distance_errors)}
        for key, value in checks.items(): compare.check(value, proposal["calibration_checks"][target][key], "calibration", key)
        models[target] = {"eigenvalue_minimum": float(eigenvalues[0]), "condition_number": float(eigenvalues[-1]/eigenvalues[0]),
                          "Newton_Mahalanobis_length": result["Newton_Mahalanobis_length"], **checks}
    return models


def run(data, physics_path, design_path, output):
    start = time.process_time(); data, output = Path(data), Path(output)
    require(not output.exists(), "never overwrite a replay receipt")
    names = ["registration.json", "report.json", "supervisor.json", "frozen-proposal.json", "diagnostic-ledger.json",
             "calibration-initial-ledger.json", "calibration-followup-ledger.json", "main-results.json", "control-results.json", "calls.jsonl"]
    inputs = [identify(data/name) for name in names]+[identify(physics_path), identify(design_path), identify(__file__)]
    loaded = {name: read(data/name) for name in names if name != "calls.jsonl"}
    report = loaded["report.json"]; supervisor = loaded["supervisor.json"]; registration = loaded["registration.json"]
    require(report["status"] == "complete_excluded_efficiency_pilot" and report["stage"] == "complete" and report["failure"] is None, "actual complete excluded report required")
    require(supervisor["worker_exit_code"] == 0 and supervisor["all_owned_workers_drained"] is True and supervisor["report_sha256"] == identify(data/"report.json")["sha256"], "actual supervisor completion identity")
    require(report["source_guards_unchanged"] is True and report["registration_sha256"] == identify(data/"registration.json")["sha256"], "original admission binding")
    for name, digest in report["ledger_sha256"].items(): require(identify(data/name)["sha256"] == digest, "saved ledger identity "+name)
    require(report["proposal_sha256"] == identify(data/"frozen-proposal.json")["sha256"], "proposal identity")
    design = read(design_path); physics = read(physics_path)["physics"]
    require(identify(design_path)["sha256"] == registration["design_sha256"] == loaded["frozen-proposal.json"]["design_sha256"], "frozen design identity")
    constants = physics["coordinate_constants"]; offset = design["physics"]["energy_offset_hartree"]
    require(design["physics"]["beads"] == 1 and tuple(design["fresh_diagnostic_draws"]["seeds"]) == SEEDS, "fixed P1 cohort")
    compare = Comparison(); events = [json.loads(line) for line in (data/"calls.jsonl").read_text().splitlines()]
    require(len(events) % 2 == 0, "incomplete attempted-call journal")
    energies = {}
    for number, (attempt, success) in enumerate(zip(events[::2], events[1::2]), 1):
        require(attempt["event"] == "attempt" and success["event"] == "success" and attempt["attempt_number"] == success["attempt_number"] == number, "ordered successful call journal")
        key = (attempt["stage"], attempt["index"])
        require(key == (success["stage"], success["index"]) and key not in energies, "unique evaluated target identity")
        require(math.isfinite(success["energy_hartree"]), "finite saved energy")
        energies[key] = success["energy_hartree"]
    require(len(energies) == report["attempted_PES_geometries"] == report["successful_PES_geometries"] <= 4000, "actual PES accounting")
    proposal = loaded["frozen-proposal.json"]
    calibration = calibration_replay(loaded["calibration-initial-ledger.json"], loaded["calibration-followup-ledger.json"], proposal, energies, constants, offset, compare)
    ledger = loaded["diagnostic-ledger.json"]; rows, controls = ledger["main_rows"], ledger["control_rows"]
    check_membership(rows, controls)
    mixtures = {False: FullMixture(proposal["models"]), True: FullMixture(proposal["models"], True)}
    reconstructed, expected_calls, zero_counts = [], set(), Counter()
    for stage, selected, result_name in (("main_diagnostics", rows, "main-results.json"), ("entry_separation_control", controls, "control-results.json")):
        saved = loaded[result_name]
        require(saved["ledger_sha256"] == identify(data/"diagnostic-ledger.json")["sha256"], "diagnostic-result ledger identity")
        require([r["index"] for r in selected] == [r["index"] for r in saved["results"]], "complete result indices")
        weights = {}
        for row, observed in zip(selected, saved["results"]):
            entry = row["target"] == "entry"; h = 3 if entry else row["channel"]
            alpha = None if entry else float(row["target"].split("_")[1])
            calculated = coordinate_measure(row["shape"], row["direction"], h, alpha, constants, row["separation_bohr"])
            for key, value in calculated.items():
                if key == "domains": require(value == row[key], "all nested domains exact")
                else: compare.check(value, row[key], "geometry", f"{stage}/{row['index']}/{key}")
            density = mixtures[entry](row["shape"], row["direction"])
            compare.check(density, row["log_proposal_density"], "density", "full mixture density")
            if calculated["common_zero_reason"] is not None or not calculated["domains"]["outer"]:
                require(observed["energy_hartree"] is None and (stage, row["index"]) not in energies, "zero draw must not gain a PES call")
                values = dict.fromkeys(DOMAINS, None); zero_counts[calculated["common_zero_reason"]] += 1
            else:
                key = stage, row["index"]; expected_calls.add(key)
                require(observed["energy_hartree"] == energies[key], "result energy equals saved call exactly")
                logweight = calculated["log_geometric_measure"]-density-constants["beta_hartree_inverse"]*(energies[key]-offset)
                values = {d: logweight if calculated["domains"][d] else None for d in DOMAINS}
            for domain in DOMAINS: compare.check(values[domain], observed["log_weight"][domain], "density", "saved log weight")
            weights[row["index"]] = values
        reconstructed.append(weights)
    expected_calls |= {("initial_calibration", i) for i in range(711)} | {("followup_calibration", i) for i in range(711, 800)}
    require(set(energies) == expected_calls and len(rows) == report["main_draws_complete"] == 3072 and len(controls) == report["control_draws_complete"] == 128, "all4000 staged targets accounted without discarded zeros")
    analysis = independent_analysis(rows, controls, *reconstructed)
    compare.tree(analysis, report["analysis"])
    require(not analysis["original_domain_efficiency_screen_passed"], "recorded failed screen must remain failed")
    for row in inputs: require(identify(row["path"]) == row, "input changed during read-only replay")
    forbidden = [name for name in sys.modules if name == "pyrpmd" or name.startswith("pyrpmd.") or name == "benchmarks" or name.startswith("benchmarks.")]
    require(not forbidden, "producer/physical-module import is outside replay scope")
    result = {"schema": "pyrpmd.hch4-efficiency-independent-calculation-replay.v1", "scope": __doc__,
              "all_reconstruction_checks_passed": True, "author_independent_of_producer": False,
              "PES_evaluations": 0, "random_draws": 0, "dynamics_steps": 0,
              "producer_analysis_called": False, "inputs_unchanged": True, "inputs": inputs,
              "comparison_tolerances": TOLERANCES, "comparison_counts": dict(compare.checked),
              "maximum_absolute_reconstruction_errors": dict(compare.maximum_absolute_error),
              "staged_geometries": {"calibration": 800, "main": 3072, "entry_control": 128},
              "actual_saved_PES_calls": len(energies), "retained_geometric_zero_draws": dict(zero_counts),
              "calibration_recomputed": calibration, "analysis_recomputed": analysis,
              "saved_execution_report": inputs[1], "replay_CPU_seconds_before_write": time.process_time()-start,
              "interpretation": "Successful reconstruction verifies the saved calculation, not physical-domain adequacy, unseen tails, rate normalization, quantum effects or paper agreement. Both surfaces remain P1. Conditional eight-seed intervals are nominal delta/t approximations."}
    with output.open("x") as stream: json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False); stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "physics", "design", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    outcome = run(args.data, args.physics, args.design, args.output)
    print(json.dumps({k: outcome[k] for k in ("all_reconstruction_checks_passed", "PES_evaluations", "random_draws", "actual_saved_PES_calls", "maximum_absolute_reconstruction_errors")}, indent=2))
