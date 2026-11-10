"""Bounded-workspace MBAR solve, using native or NumPy block reductions."""
from __future__ import annotations

import os
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.sparse.csgraph import connected_components

from .mbar_problem import ReducedPotentials


class Reduction:
    def __init__(self, problem, counts, target, backend, threads):
        self.problem, self.counts, self.target = problem, np.asarray(counts, dtype=float), target
        self.native = None
        if backend != "numpy":
            try:
                from .. import _mbar
                self.native = _mbar
            except ImportError:
                if backend == "native":
                    raise RuntimeError("native MBAR extension is unavailable; build pyrpmd._mbar") from None
        available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
        self.threads = min(threads, available)
        self.observed_threads = 1
        self.calls = 0

    def __call__(self, free, mode=0):
        self.calls += 1
        p = self.problem
        k, n = p.shape
        if self.native is not None:
            result = self.native.reduce(p.values, p.centers, p.stiffness, p.periods,
                                        p.rows, p.indices, np.ascontiguousarray(free), self.counts,
                                        self.target, p.beta, mode, self.threads)
            new, gradient, matrix, logweights, state_weights = [
                np.frombuffer(value, dtype=np.float64) for value in result[:5]]
            self.observed_threads = max(self.observed_threads, result[6])
            return (new, result[5], gradient, matrix.reshape((k, k)) if mode else None,
                    logweights, state_weights.reshape((n, k)) if mode == 3 else None)
        logn = np.log(self.counts)
        totals = np.zeros(k)
        logsum = np.full(k, -np.inf)
        matrix = np.zeros((k, k)) if mode else None
        logweights = np.empty(n) if mode >= 2 else None
        state_weights = np.empty((n, k)) if mode == 3 else None
        objective = 0.
        for start in range(0, n, 1024):
            stop = min(n, start + 1024)
            u = p.block(start, stop)
            if not np.isfinite(u).all():
                raise ValueError("nonfinite MBAR reduced potentials")
            common = u.min(axis=0)
            u -= common
            logd = logsumexp(logn[:, None] + free[:, None] - u, axis=0)
            logsum = np.logaddexp(logsum, logsumexp(-u-logd, axis=1))
            probabilities = np.exp(logn[:, None] + free[:, None] - u - logd)
            totals += probabilities.sum(axis=1)
            objective += logd.sum()
            if mode:
                if mode >= 2:
                    probabilities /= self.counts[:, None]
                    logweights[start:stop] = -self.target[start:stop] + common - logd
                matrix += probabilities @ probabilities.T
                if mode == 3:
                    state_weights[start:stop] = probabilities.T
        new = -logsum
        new -= new[0]
        if mode == 1:
            matrix = np.diag(totals) - matrix
        return new, float(objective-self.counts@free), totals-self.counts, matrix, logweights, state_weights


def solve(problem, counts, target, initial, *, tolerance, max_iterations,
          overlap_threshold, combine_identical_states, return_state_weights, backend, threads):
    from .mbar import MBARResult
    if not isinstance(problem, ReducedPotentials):
        problem = ReducedPotentials(problem)
    representatives, state_map = (problem.representatives() if combine_identical_states else
                                   (np.arange(len(counts)), np.arange(len(counts))))
    original_counts = counts
    counts = np.bincount(state_map, weights=counts).astype(int)
    reduced = problem.select(rows=representatives)
    f = initial[representatives].copy()
    f -= f[0]
    evaluate = Reduction(reduced, counts, target, backend, threads)
    k = len(counts)
    for _ in range(min(8, max_iterations)):
        new = evaluate(f)[0]
        if np.max(abs(new-f)) < tolerance:
            break
        f = new
    newton_iterations = lbfgs_iterations = 0
    fallback = False
    if k > 1:
        for newton_iterations in range(1, min(80, max_iterations)+1):
            new, value, gradient, hessian, _, _ = evaluate(f, 1)
            residual = float(np.max(abs(new-f)))
            if residual < tolerance:
                break
            try:
                step = np.linalg.solve(hessian[1:, 1:], gradient[1:])
            except np.linalg.LinAlgError:
                fallback = True
                break
            descent = float(gradient[1:] @ step)
            if not np.isfinite(step).all() or descent <= 0:
                fallback = True
                break
            alpha = min(1., 20./max(float(np.max(abs(step))), 1e-30))
            rounding = 16*np.finfo(float).eps*max(1., abs(value))
            for _ in range(24):
                candidate = f.copy()
                candidate[1:] -= alpha*step
                proposal, candidate_value, _, _, _, _ = evaluate(candidate)
                better = candidate_value <= value - 1e-4*alpha*descent
                near = candidate_value <= value+rounding and np.max(abs(proposal-candidate)) < residual
                if np.isfinite(candidate_value) and (better or near):
                    f = candidate
                    break
                alpha *= .5
            else:
                fallback = True
                break
        if np.max(abs(evaluate(f)[0]-f)) >= tolerance:
            fallback = True
    if k > 1 and fallback and np.max(abs(evaluate(f)[0]-f)) >= tolerance:
        def objective(x):
            result = evaluate(np.r_[0., x])
            return result[1], result[2][1:]
        fit = minimize(objective, f[1:], method="L-BFGS-B", jac=True,
                       options={"ftol": 1e-14, "gtol": tolerance, "maxiter": max_iterations, "maxls": 40})
        f = np.r_[0., fit.x]
        lbfgs_iterations = int(fit.nit)
    for polish_iterations in range(max_iterations):
        new = evaluate(f)[0]
        if np.max(abs(new-f)) < tolerance:
            break
        f = new
    new, _, _, gram, logw, states = evaluate(f, 3 if return_state_weights else 2)
    residual = float(np.max(abs(new-f)))
    logw = logw-logsumexp(logw)
    weights = np.exp(logw)
    overlap = gram[np.ix_(state_map,state_map)] * original_counts[None]
    _, components = connected_components(np.maximum(overlap,overlap.T)>overlap_threshold, directed=False)
    diagnostics = {
        "method": "single_state" if k==1 else "lbfgs_fixed_point" if fallback else "damped_newton",
        "unique_states": k, "original_states": len(original_counts), "state_to_unique": state_map,
        "newton_iterations": newton_iterations, "lbfgs_iterations": lbfgs_iterations,
        "polish_iterations": polish_iterations,
        "backend": "native" if evaluate.native is not None else "numpy_blocked",
        "threads_requested": threads, "threads_used": evaluate.observed_threads,
        "native_calls": evaluate.calls if evaluate.native is not None else 0,
        "potential_storage": "harmonic_on_demand" if len(problem.centers) else "dense_input_view",
        "state_weights_materialized": bool(return_state_weights),
        "workspace_scaling": "O(N + threads*K*K); excludes requested N*K state_weights and caller input",
    }
    return MBARResult(f[state_map], weights, states[:, state_map] if return_state_weights else None,
                      overlap, components, residual<tolerance, residual, diagnostics, logw)
