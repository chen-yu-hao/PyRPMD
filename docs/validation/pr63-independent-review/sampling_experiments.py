"""Independent equation-level review of PR 63 enhanced-sampling methods.

Run from the repository root with PYTHONPATH=src python <this file>.
References and the distinction between numerical checks and paper reproduction
are recorded in sampling_review.md. No existing benchmark outputs are reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t

from pyrpmd.advanced_cv import CoordinationNumber, Dihedral, PathCV, RMSD
from pyrpmd.biased_kinetics import estimate_biased_rate, rescale_biased_time
from pyrpmd.cv import Cartesian, CVStack
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.opes import OPESBias
from pyrpmd.potentials.base import PESResult, Potential
from pyrpmd.potentials.builtin import HarmonicPotential
from pyrpmd.replica_exchange import ReplicaExchange
from pyrpmd.retis import ReplicaExchangeTIS
from pyrpmd.shared_metadynamics import SharedMetadynamicsBias


OUT = Path(__file__).resolve().parent
SEED = 638261


def derivatives():
    """Fourth-order finite differences of full-ring CVs and invariances."""
    rng = np.random.default_rng(SEED)
    rows = []
    reference = rng.normal(size=(5, 3))
    refs = np.stack([reference + rng.normal(scale=0.35, size=(5, 3)) for _ in range(4)])
    for beads in (1, 5):
        for estimator in ("centroid", "bead_average"):
            q = reference[None] + rng.normal(scale=0.25, size=(beads, 5, 3))
            cvs = [Dihedral((0, 1, 2, 3), estimator=estimator),
                   CoordinationNumber((0, 1), (2, 3, 4), r0=1.2, estimator=estimator),
                   RMSD(reference, weights=[1, 2, 3, 4, 5], estimator=estimator),
                   PathCV(refs, path_lambda=2.5, scale=1.3, estimator=estimator)]
            for cv in cvs:
                analytic = cv.value_and_jacobian(q)
                numerical = np.zeros_like(analytic.jacobian)
                h = 1e-5
                for index in np.ndindex(q.shape):
                    delta = np.zeros_like(q)
                    delta[index] = h
                    values = [cv.value_and_jacobian(q + j * delta).value for j in (-2, -1, 1, 2)]
                    numerical[(slice(None),) + index] = (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * h)
                translation = np.max(np.abs(cv.value_and_jacobian(q + [1.7, -0.3, 2.2]).value - analytic.value))
                u, _, vt = np.linalg.svd(rng.normal(size=(3, 3)))
                rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
                # Geometric CVs are invariant to a rigid rotation of the query.
                rotated = cv.value_and_jacobian(q @ rotation)
                rotation_error = float(np.max(np.abs(rotated.value - analytic.value)))
                error = float(np.max(np.abs(numerical - analytic.jacobian)))
                assert error < 2e-8, (type(cv).__name__, error)
                assert translation < 1e-12 and rotation_error < 1e-12
                rows.append(dict(cv=type(cv).__name__, beads=beads, estimator=estimator,
                                 derivative_max_abs=error, translation_max_abs=float(translation),
                                 rotation_max_abs=rotation_error))
    # Circular bead mean crosses the atan2 seam smoothly; use angle differences.
    def torsion_q(angle):
        return np.array([[1., 0, 0], [0., 0, 0], [0., 0, 1], [np.cos(angle), np.sin(angle), 1.]])
    q = np.stack([torsion_q(np.pi - 0.01), torsion_q(-np.pi + 0.01)])
    circ = Dihedral((0, 1, 2, 3), estimator="bead_average").value_and_jacobian(q)
    assert abs(abs(circ.value[0]) - np.pi) < 1e-12
    return rows


def mixture(points, centers, widths, weights, periods):
    """Independent direct Gaussian-image sum; no production OPES helpers."""
    pts = np.atleast_2d(points)
    kernels = np.ones((len(pts), len(centers)))
    scores = np.zeros((len(pts), len(centers), pts.shape[1]))
    for dim, period in enumerate(periods):
        delta = pts[:, None, dim] - centers[None, :, dim]
        width = widths[None, :, dim]
        shifts = np.array([0.0]) if period is None else np.arange(-16, 17) * period
        dx = delta[..., None] + shifts
        image = np.exp(-0.5 * (dx / width[..., None]) ** 2) / (np.sqrt(2 * np.pi) * width[..., None])
        dens = image.sum(axis=-1)
        score = (image * (-dx / width[..., None] ** 2)).sum(axis=-1) / dens
        kernels *= dens
        scores[..., dim] = score
    contribution = kernels * weights
    probability = contribution.sum(axis=1) / np.sum(weights)
    gradient = (contribution[..., None] * scores).sum(axis=1) / np.sum(weights)
    return probability, gradient


def opes_equations():
    rows = []
    cv = CVStack([Cartesian(0, 0), Cartesian(0, 1)])
    # A periodic first component also exercises the Fourier/image crossover.
    cv.periods = (2 * np.pi, None)
    centers = np.array([[2.9, -0.3], [-3.0, 0.8], [2.0, -0.7], [0.7, 0.2], [-0.5, 1.2], [-2.8, -0.5]])
    points = np.array([[-3.1, -0.6], [-1.2, 0.3], [0.0, 0.7], [2.8, 0.3]])
    initial = np.array([1.3, 0.7])
    beta, barrier, gamma = 1.7, 4.0, 6.0
    for mode in ("metad", "explore"):
        for adaptive in (False, True):
            bias = OPESBias(cv, initial, beta, barrier=barrier, bias_factor=gamma,
                            mode=mode, adaptive_sigma=adaptive, sigma_min=initial * 0.05 if adaptive else None)
            weights, widths, unwrapped = [], [], []
            prefactor = ((gamma - 1) if mode == "explore" else (1 - 1 / gamma)) / beta
            eps = np.exp(-barrier / prefactor)
            largest_energy = largest_gradient = largest_z = largest_width = 0.0
            for index, center in enumerate(centers):
                if index == 0:
                    before = -barrier
                else:
                    prob, _ = mixture(center, centers[:index], np.array(widths), np.array(weights), cv.periods)
                    pcenters, _ = mixture(centers[:index], centers[:index], np.array(widths), np.array(weights), cv.periods)
                    before = prefactor * np.log(prob[0] / pcenters.mean() + eps)
                weights.append(1.0 if mode == "explore" else np.exp(beta * before))
                if not unwrapped:
                    unwrapped.append(center.copy())
                else:
                    delta = center - centers[index - 1]
                    delta[0] = (delta[0] + np.pi) % (2 * np.pi) - np.pi
                    unwrapped.append(unwrapped[-1] + delta)
                width = initial.copy()
                if adaptive and index >= 2:
                    w = np.array(weights)
                    neff = w.sum() ** 2 / (w @ w)
                    width = np.maximum(np.std(unwrapped, axis=0, ddof=1) * neff ** (-1 / 6), initial * 0.05)
                widths.append(width)
                bias.update(center, step=index + 1, time=(index + 1) * 0.01)
                density, derivative = mixture(points, centers[:index + 1], np.array(widths), np.array(weights), cv.periods)
                at_centers, _ = mixture(centers[:index + 1], centers[:index + 1], np.array(widths), np.array(weights), cv.periods)
                z = at_centers.mean()
                expected_energy = prefactor * np.log(density / z + eps)
                expected_gradient = prefactor * derivative / (density + eps * z)[:, None]
                actual_energy, actual_gradient = bias.physical_energy_gradient(points)
                largest_energy = max(largest_energy, float(np.max(np.abs(actual_energy - expected_energy))))
                largest_gradient = max(largest_gradient, float(np.max(np.abs(actual_gradient - expected_gradient))))
                largest_z = max(largest_z, abs(bias.log_z - np.log(z)))
                if adaptive:
                    largest_width = max(largest_width, float(np.max(np.abs(bias.state_dict()["kernel_sigmas"] - np.array(widths)))))
            assert largest_energy < 2e-12 and largest_gradient < 2e-12 and largest_z < 2e-12
            assert largest_width < 2e-12
            restored = OPESBias(cv, initial, beta, barrier=barrier, bias_factor=gamma,
                                mode=mode, adaptive_sigma=adaptive,
                                sigma_min=initial * 0.05 if adaptive else None, state=bias.state_dict())
            assert np.array_equal(restored.physical_energy_gradient(points)[0], bias.physical_energy_gradient(points)[0])
            rows.append(dict(mode=mode, adaptive=adaptive, updates=len(centers), max_energy_error=largest_energy,
                             max_gradient_error=largest_gradient, max_log_z_error=largest_z, max_width_error=largest_width))
    return rows


def exchange_equations():
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for beads in (1, 2, 8, 32):
        for hamiltonian in (False, True):
            betas = [0.8, 2.3]
            ks = [0.7, 1.6 if hamiltonian else 0.7]
            masses = np.array([1.2, 2.7])
            dyns = [Dynamics(HarmonicPotential(k), masses, beta, beads, 0.02, native=False)
                    for k, beta in zip(ks, betas)]
            qs = [rng.normal(scale=0.3, size=(beads, 2, 3)) for _ in betas]
            states = [State(q, rng.normal(size=q.shape)) for q in qs]
            exchange = ReplicaExchange(dyns, states, seed=SEED)
            def reduced(beta, k, q):
                spring = 0.5 * np.sum(masses[None, :, None] * (beads / beta) ** 2 * (q - np.roll(q, -1, axis=0)) ** 2)
                return beta / beads * (0.5 * k * np.sum(q * q) + spring)
            u00, u11 = reduced(betas[0], ks[0], qs[0]), reduced(betas[1], ks[1], qs[1])
            u01, u10 = reduced(betas[0], ks[0], qs[1]), reduced(betas[1], ks[1], qs[0])
            exact = u00 + u11 - u01 - u10
            measured = exchange.log_acceptance_ratio(0, 1)
            error = abs(exact - measured)
            # Accepted energy-weighted forward/reverse fluxes must agree.
            forward = -(u00 + u11) + min(0.0, measured)
            reverse = -(u01 + u10) + min(0.0, -measured)
            balance = abs(forward - reverse)
            assert error < 1e-10 and balance < 1e-10
            # Equal configurations guarantee acceptance, isolating temperature momentum mapping.
            exchange.states[1].q[:] = exchange.states[0].q
            old_p = [s.p.copy() for s in exchange.states]
            old_kinetic = sum(beta / beads * np.sum(p * p / (2 * masses[None, :, None])) for beta, p in zip(betas, old_p))
            assert exchange.attempt(0, 1).accepted
            new_kinetic = sum(beta / beads * np.sum(s.p * s.p / (2 * masses[None, :, None])) for beta, s in zip(betas, exchange.states))
            assert abs(new_kinetic - old_kinetic) < 1e-12
            rows.append(dict(beads=beads, hamiltonian=hamiltonian, log_ratio_error=error,
                             log_detailed_balance_error=balance, kinetic_reduced_error=abs(new_kinetic - old_kinetic)))
    return rows


def shared_metadynamics():
    cv = Cartesian(0)
    bias = SharedMetadynamicsBias(cv, sigma=[0.4], height=0.3, mode="well_tempered",
                                 beta=1.8, bias_factor=5.0, walkers=4)
    first = np.array([[-0.8], [-0.2], [0.4], [0.9]])
    second = np.array([[-0.5], [0.1], [0.6], [0.8]])
    bias.deposit_walkers(first, step=10, time=0.1)
    before = 0.3 * np.exp(-0.5 * ((second - first.T) / 0.4) ** 2).sum(axis=1)
    expected = 0.3 * np.exp(-1.8 * before / 4)
    batch = bias.deposit_walkers(second, step=20, time=0.2)
    heights = np.array([hill["height"] for hill in batch])
    assert np.max(np.abs(heights - expected)) < 2e-15
    return dict(walkers=4, batches=2, height_max_abs=float(np.max(np.abs(heights - expected))))


def kinetics():
    boundaries = np.array([0., .2, .7, 1.9, 2.0])
    biases = np.array([1., -0.5, 0.7, 1.2])
    reference = np.array([-0.3, -0.1, 0.2, 0.4])
    beta = 1.4
    exact = float(np.dot(np.diff(boundaries), np.exp(beta * (biases - reference))))
    result = rescale_biased_time(boundaries, biases, beta=beta, transition_bias=reference, transition_unbiased=True)
    gauge = rescale_biased_time(boundaries, biases + 37.0, beta=beta, transition_bias=reference + 37.0, transition_unbiased=True)
    assert abs(result["rescaled_time_au"] - exact) < 1e-12
    assert abs(gauge["rescaled_time_au"] - exact) < 1e-12
    rng = np.random.default_rng(SEED + 2)
    rate = 0.7
    event_times = rng.exponential(1 / rate, size=40000)
    censor_times = rng.uniform(0.3, 3, size=len(event_times))
    events = event_times <= censor_times
    exposure = np.minimum(event_times, censor_times)
    fit = estimate_biased_rate(exposure, events=events, bootstrap=0)
    sem = np.sqrt(fit["events"]) / np.sum(exposure)
    z = (fit["rate_au"] - rate) / sem
    return dict(clock_exact=exact, clock_max_error=abs(result["rescaled_time_au"] - exact),
                gauge_max_error=abs(gauge["rescaled_time_au"] - exact),
                censoring=fit, known_rate_au=rate, rate_sem_asymptotic=sem, rate_standardized_error=z,
                statistical_note="One predeclared seed; asymptotic uncertainty, no molecular kinetics claim")


class Barrier(Potential):
    """Unit double well along x; independent confining y/z harmonics."""
    metadata = {"provider": "independent_review_barrier"}

    def evaluate(self, q):
        x = q[..., 0]
        energy = np.sum((x * x - 1) ** 2 + 0.5 * np.sum(q[..., 1:] ** 2, axis=-1), axis=1)
        forces = -q.copy()
        forces[..., 0] = -4 * x * (x * x - 1)
        return PESResult(energy, forces)


def retis(samples, seeds, dt):
    beta = 1.3
    interfaces = np.array([-0.8, -0.5, -0.2])
    energies = (interfaces * interfaces - 1) ** 2
    reference = np.exp(-beta * np.diff(np.r_[energies, 1.0]))
    raw, rows = {}, []
    for seed in seeds:
        dyn = Dynamics(Barrier(), [1.0], beta, 1, dt, seed=seed, native=False)
        sampler = ReplicaExchangeTIS(dyn, Cartesian(0), basin_a=-0.8, basin_b=0.8,
                                     interfaces=interfaces, seed=seed, max_path_steps=2500)
        q = np.zeros((1, 1, 3))
        p = np.zeros_like(q)
        p[0, 0, 0] = 1.0
        initial = sampler.generate_path(State(q, p), 2)
        start = time.monotonic()
        result = sampler.sample_interfaces([initial] * len(interfaces), n_samples=samples, burn_in=max(100, samples // 4))
        crossing = np.stack([item.crossing for item in result.interfaces], axis=1)
        raw[f"crossing_{seed}"] = crossing
        raw[f"lengths_{seed}"] = np.stack([item.path_frames for item in result.interfaces], axis=1)
        raw[f"labels_{seed}"] = result.walker_labels
        measured = crossing.mean(axis=0)
        row = dict(seed=seed, samples=samples, burn_in=max(100, samples // 4), dt=dt,
                   probabilities=measured.tolist(), product=float(np.prod(measured)),
                   length_rejections=sum(item.length_rejections for item in result.interfaces),
                   swaps=result.swap_accepts.tolist(), attempts=result.swap_attempts.tolist(),
                   seconds=time.monotonic() - start)
        rows.append(row)
        print("RETIS", json.dumps(row), flush=True)
    products = np.array([row["product"] for row in rows])
    mean = products.mean()
    ci = None if len(seeds) < 2 else float(t.ppf(0.975, len(seeds) - 1) * products.std(ddof=1) / np.sqrt(len(seeds)))
    np.savez_compressed(OUT / f"sampling_retis_raw_dt{dt}.npz", **raw)
    return dict(beta=beta, interfaces=interfaces.tolist(), basin_a=-0.8, basin_b=0.8,
                continuum_reference_probabilities=reference.tolist(), continuum_reference_product=float(np.prod(reference)),
                finite_dt_warning="Continuum flux reference; finite integration and path-bound errors remain to be quantified",
                samples_per_seed=samples, seeds=rows, mean_product=float(mean), seed_t_95_halfwidth=ci)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retis-samples", type=int, default=600)
    parser.add_argument("--retis-seeds", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.04)
    args = parser.parse_args()
    started = time.monotonic()
    root = OUT.parents[2]
    files = ["advanced_cv.py", "opes.py", "replica_exchange.py", "biased_kinetics.py", "retis.py", "transition_interface.py", "shared_metadynamics.py"]
    def hashes():
        return {f"src/pyrpmd/{p}": hashlib.sha256((root / "src" / "pyrpmd" / p).read_bytes()).hexdigest() for p in files}
    before = hashes()
    result = dict(head_reviewed="3faac3aa42b5424b2711367608e0138c82f62e0a", seed=SEED,
                  runtime=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  cv=derivatives(), opes=opes_equations(), replica_exchange=exchange_equations(),
                  shared_metadynamics=shared_metadynamics(), kinetics=kinetics())
    print("deterministic checks completed", flush=True)
    if args.retis_samples:
        result["retis"] = retis(args.retis_samples, [SEED + 100 + i for i in range(args.retis_seeds)], args.dt)
    result["seconds"] = time.monotonic() - started
    assert hashes() == before, "Production source changed during the numerical run"
    result["source_sha256"] = before
    result["driver_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    filename = OUT / f"sampling_results_dt{args.dt}.json"
    filename.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(str(filename), flush=True)


if __name__ == "__main__":
    main()
