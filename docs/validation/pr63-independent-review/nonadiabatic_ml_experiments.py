"""Independent direct-data transport and ML/RPC review experiments.

Run from the repository root with PYTHONPATH=src.  References and scope are
recorded in nonadiabatic_ml_report.md; this is not a molecular QC validation.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from pyrpmd.adiabatic import (
    AdiabaticPythonPotential, AdiabaticResult, AdiabaticRPSHDynamics,
    ElectronicFrame, QDNRPMDDynamics,
)
from pyrpmd.dynamics import State
from pyrpmd.mash import MASHDynamics
from pyrpmd.nonadiabatic import DiabaticResult, NRPMDDynamics, NRPMDState
from pyrpmd.potentials.base import PESResult
from pyrpmd.potentials.lifecycle import CommitteePotential, DeltaLearningPotential


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent / "nonadiabatic_ml"
SEED = 624918


class ThreeStateDiabatic:
    """Hermitian 3-state model with two nuclear coordinates and complex mixing."""
    def __init__(self):
        self.h0 = np.diag([-.7, .2, 1.1]).astype(complex)
        self.a = np.array([[.12, .2+.07j, -.04j], [.2-.07j, -.08, .11],
                           [.04j, .11, -.04]])
        self.b = np.array([[.04, -.06j, .14], [.06j, -.1, .08+.03j],
                           [.14, .08-.03j, .06]])
        self.c = np.array([[.1, .04, 0], [.04, -.06, .02j], [0, -.02j, -.04]])

    def evaluate(self, q):
        x, y = q[:, 0, 0], q[:, 1, 1]
        scalar = .25 * np.sum(q*q, axis=(1, 2))
        h = (self.h0 + scalar[:, None, None]*np.eye(3)
             + x[:, None, None]*self.a + y[:, None, None]*self.b
             + .2*x[:, None, None]**2*self.c)
        g = np.einsum("ij,bac->bijac", np.eye(3), .5*q).astype(complex)
        g[:, :, :, 0, 0] += self.a + .4*x[:, None, None]*self.c
        g[:, :, :, 1, 1] += self.b
        return DiabaticResult(h, g)


class ThreeStateQC:
    def __init__(self, model, gauge):
        self.model, self.gauge, self.calls = model, gauge, 0

    def evaluate(self, q, previous_q=None, previous_context=None):
        h = self.model.evaluate(q)
        e, u = np.linalg.eigh(h.matrix)
        self.calls += 1
        if self.gauge:
            angles = (np.array([.31, -.73, 1.27]) * self.calls
                      + q[:, 0, 0, None] * np.array([.2, .5, -.4]))
            u = u * np.exp(1j*angles)[:, None, :]
        g = np.einsum("bsi,bstac,btj->bijac", u.conj(), h.gradients, u)
        nac = np.zeros_like(g)
        for i in range(3):
            for j in range(3):
                if i != j:
                    nac[:, i, j] = g[:, i, j] / (e[:, j]-e[:, i])[:, None, None]
        overlaps = None if previous_context is None else previous_context.conj().swapaxes(1, 2) @ u
        return AdiabaticResult(e, g[:, np.arange(3), np.arange(3)].real, nac,
                               overlaps, context=u)


def representation_equivalence():
    rng = np.random.default_rng(SEED)
    model = ThreeStateDiabatic()
    rows, raw = [], {}
    for beads in (1, 4):
        q = rng.normal(scale=.2, size=(beads, 2, 3))
        p = rng.normal(scale=.1, size=q.shape)
        for gauge in (False, True):
            for dt in (.04, .02, .01):
                kwargs = dict(masses=[1.7, 2.3], beta=3., nbeads=beads,
                              dt=dt, nstates=3, native=False)
                qc = ThreeStateQC(model, gauge)
                qd = QDNRPMDDynamics(AdiabaticPythonPotential(qc), **kwargs)
                direct = NRPMDDynamics(model, **kwargs)
                ad = qd.initialize(State(q, p), seed=SEED+beads)
                z = np.einsum("bij,bj->bi", ad.electronic_frame.result.context,
                              ad.mapping_q+1j*ad.mapping_p)
                fixed = NRPMDState(q, p, mapping_q=z.real, mapping_p=z.imag)
                errors, trajectory = [], []
                for _ in range(round(.8/dt)):
                    qd.step(ad)
                    direct.step(fixed)
                    transformed = np.einsum("bij,bj->bi", ad.electronic_frame.result.context,
                                           ad.mapping_q+1j*ad.mapping_p)
                    errors.append([np.max(abs(ad.q-fixed.q)), np.max(abs(ad.p-fixed.p)),
                                   np.max(abs(transformed-fixed.mapping_q-1j*fixed.mapping_p)),
                                   abs(qd.energy(ad)-direct.energy(fixed))])
                    trajectory.append([*ad.q.ravel(), *ad.p.ravel(), *transformed.real.ravel(),
                                       *transformed.imag.ravel()])
                maximum = np.max(errors, axis=0)
                assert np.max(maximum) < 2e-11, (beads, gauge, dt, maximum)
                rows.append(dict(beads=beads, complex_gauge=gauge, dt=dt,
                                 coordinate_error=float(maximum[0]), momentum_error=float(maximum[1]),
                                 mapping_error=float(maximum[2]), energy_error=float(maximum[3])))
                raw[f"p{beads}_g{int(gauge)}_dt{dt}"] = np.asarray(trajectory)
    np.savez_compressed(OUTPUT/"qd_trajectories.npz", **raw)
    return rows


GENERATOR = np.array([[0., -1.], [1., 0.]])
ENERGIES = np.array([.1, .8])


def prescribed_frame(t, previous=None):
    q = np.array([[[t, 0., 0.]]])
    theta = .15*t + .05*t*t
    u = expm(theta*GENERATOR)
    nac = GENERATOR * (.15+.1*t)
    d = np.zeros((1, 2, 2, 1, 3))
    d[0, :, :, 0, 0] = nac
    overlap = None if previous is None else (previous.result.context.T@u)[None]
    result = AdiabaticResult(ENERGIES[None], np.zeros((1, 2, 1, 3)), d,
                             overlap, context=u)
    return ElectronicFrame(q, result)


def electronic_convergence():
    c0 = np.array([np.sqrt(.8), np.sqrt(.2)*1j])
    rhs = lambda t, c: (-1j*np.diag(ENERGIES)-(.15+.1*t)*GENERATOR) @ c
    reference = solve_ivp(rhs, (0., 4.), c0, method="DOP853", rtol=2e-13,
                          atol=2e-14, dense_output=True)
    tighter = solve_ivp(rhs, (0., 4.), c0, method="DOP853", rtol=3e-14, atol=3e-15)
    assert reference.success and tighter.success
    rows = []
    for dt in (.4, .2, .1, .05, .025):
        dyn = AdiabaticRPSHDynamics(None, [1.], 1., 1, dt, 2, native=False,
                                    electronic_substeps=4)
        old = prescribed_frame(0.)
        c = c0.copy()
        maximum = 0.
        for k in range(round(4./dt)):
            new = prescribed_frame((k+1)*dt, old)
            c, _, _ = dyn._electronic_step(c, old, new, int(np.argmax(abs(c)**2)))
            maximum = max(maximum, float(np.linalg.norm(c-reference.sol((k+1)*dt))))
            old = new
        rows.append(dict(dt=dt, coefficient_error=float(np.linalg.norm(c-reference.y[:, -1])),
                         maximum_path_error=maximum, norm_error=float(abs(np.vdot(c,c)-1))))
    for i in range(1, len(rows)):
        rows[i]["observed_order"] = float(np.log2(rows[i-1]["coefficient_error"]/rows[i]["coefficient_error"]))
    assert all(1.95 < r["observed_order"] < 2.05 for r in rows[1:])
    return {"reference_tolerance_change": float(np.linalg.norm(reference.y[:, -1]-tighter.y[:, -1])),
            "rows": rows}


class RotatingTwoStateQC:
    """Constant eigenvalues and constant NAC: exact piecewise event solution."""
    def evaluate(self, q, previous_q=None, previous_context=None):
        u = np.asarray([expm(.7*x*GENERATOR) for x in q[:, 0, 0]])
        nac = np.zeros((len(q), 2, 2, 1, 3))
        nac[:, :, :, 0, 0] = .7*GENERATOR
        overlap = None if previous_context is None else previous_context.swapaxes(1, 2)@u
        return AdiabaticResult(np.tile([0., .3], (len(q), 1)),
                              np.zeros((len(q), 2, 1, 3)), nac, overlap, context=u)


def mash_event_convergence():
    # On each constant-velocity segment, c(t)=exp[-i(E-i v d)t]c(0).
    # With c(0)=(1,0), the first upper-population hemisphere crossing has an
    # analytic root.  Rescale p there and propagate the remaining segment.
    e = np.diag([0., .3])
    d = .7*GENERATOR
    radius = np.sqrt(.7**2+.15**2)
    crossing = np.arcsin(np.sqrt(.5*radius**2/.7**2))/radius
    final_time = 2*crossing
    c0 = np.array([1., 0.], dtype=complex)
    at_crossing = expm(-1j*(e-1j*d)*crossing)@c0
    outgoing = np.sqrt(1.-2*.3)
    exact_c = expm(-1j*(e-1j*outgoing*d)*(final_time-crossing))@at_crossing
    exact_x = crossing + outgoing*(final_time-crossing)
    assert abs(abs(at_crossing[0])**2-.5) < 2e-15
    assert abs(exact_c[1])**2 > .5
    rows = []
    for n in (4, 8, 16, 32, 64, 128):
        # Place the exact event at a half step in every refinement: endpoint
        # delay is dt/2, and final_time/dt is the integer 2*n+1.
        dt = crossing/(n+.5)
        dyn = MASHDynamics(AdiabaticPythonPotential(RotatingTwoStateQC()),
                           [1.], 1., 1, dt, native=False)
        state = dyn.initialize(State(np.zeros((1,1,3)), np.array([[[1.,0.,0.]]])),
                               adiabatic_coefficients=c0)
        hops = 0
        for _ in range(2*n+1):
            dyn.step(state)
            hops += int(dyn.last_hop is not None and dyn.last_hop.accepted)
        assert hops == 1 and state.active_surface == 1
        error = float(abs(state.q[0,0,0]-exact_x))
        prediction = (1-outgoing)*dt/2
        assert abs(error-prediction) < 2e-12
        rows.append(dict(dt=dt, position_error=error,
                         exact_endpoint_delay_prediction=prediction,
                         coefficient_error=float(np.linalg.norm(state.coefficients-exact_c)),
                         energy_error=float(abs(dyn.energy(state)-.5)), hops=hops))
    observed_order = float(np.polyfit(np.log([r["dt"] for r in rows]),
                                      np.log([r["position_error"] for r in rows]), 1)[0])
    return {"crossing_time": float(crossing), "final_time": float(final_time),
            "exact_position": float(exact_x), "outgoing_momentum": float(outgoing),
            "position_convergence_order": observed_order, "rows": rows,
            "interpretation": "Documented endpoint MASH has first-order switched-trajectory error; energy conservation alone does not certify accuracy."}


class PolynomialPotential:
    def __init__(self, quadratic, quartic):
        self.quadratic, self.quartic = quadratic, quartic
    def evaluate(self, q):
        return PESResult(np.sum(.5*self.quadratic*q*q+self.quartic*q**4, axis=(1,2)),
                         -self.quadratic*q-4*self.quartic*q**3)


def rpc_and_committee():
    rng = np.random.default_rng(SEED+17)
    rows = []
    for beads in (4, 7, 8):
        q = rng.normal(scale=.4, size=(beads, 2, 3))
        full = DeltaLearningPotential(PolynomialPotential(.7, 0.), PolynomialPotential(.1, .03))
        for contracted in (1, 3, beads):
            split = full.split(beads, contracted)
            result = split.evaluate(q)
            fd = np.empty_like(q)
            h = 2e-6
            for index in np.ndindex(q.shape):
                plus, minus = q.copy(), q.copy()
                plus[index] += h
                minus[index] -= h
                fd[index] = -(split.evaluate(plus).energy.sum()-split.evaluate(minus).energy.sum())/(2*h)
            error = float(np.max(abs(result.forces-fd)))
            assert error < 2e-9
            full_error = float(np.max(abs(result.forces-full.evaluate(q).forces)))
            if contracted == beads:
                assert full_error == 0.
            rows.append(dict(beads=beads, contracted_beads=contracted,
                             finite_difference_force_error=error,
                             difference_from_full_force=full_error))
    # Identical misspecified models have zero disagreement despite nonzero error.
    blind = CommitteePotential([PolynomialPotential(.8, 0.), PolynomialPotential(.8, 0.)])
    q = np.array([[[2., 0., 0.]]])
    result = blind.evaluate_uncertainty(q)
    truth = PolynomialPotential(.8, .03).evaluate(q)
    return {"rpc_rows": rows, "common_bias_witness": {
        "force_disagreement": float(result.max_force_std[0]),
        "absolute_force_error": float(np.max(abs(result.mean.forces-truth.forces))),
        "interpretation": "Committee disagreement is not a calibrated error bound."}}


def main():
    OUTPUT.mkdir(exist_ok=True)
    start = time.perf_counter()
    report = {
        "source_commit_under_review": "3faac3aa42b5424b2711367608e0138c82f62e0a",
        "seed": SEED, "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "representation_equivalence": representation_equivalence(),
        "prescribed_path_electronic_convergence": electronic_convergence(),
        "mash_exact_event_comparison": mash_event_convergence(),
        "ml_and_rpc": rpc_and_committee(),
        "limitations": ["Analytic electronic providers, not quantum-chemistry software.",
                        "Representation equivalence shares MInt, not an independent physical accuracy claim.",
                        "Electronic-path ODE holds the nuclear path fixed and excludes hopping errors.",
                        "RPC tests the contracted Hamiltonian derivative, not contraction accuracy.",
                        "Single-event MASH comparison is deterministic, not a population ensemble or rate.",
                        "No new many-trajectory MASH rate or molecular benchmark is claimed."],
    }
    report["elapsed_seconds"] = time.perf_counter()-start
    paths = [Path(__file__), ROOT/"src/pyrpmd/adiabatic.py", ROOT/"src/pyrpmd/nonadiabatic.py", ROOT/"src/pyrpmd/mash.py",
             ROOT/"src/pyrpmd/potentials/contraction.py", ROOT/"src/pyrpmd/potentials/lifecycle.py"]
    report["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUTPUT/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    for name, rows in [("representation", report["representation_equivalence"]),
                       ("electronic_convergence", report["prescribed_path_electronic_convergence"]["rows"]),
                       ("mash_event_convergence", report["mash_exact_event_comparison"]["rows"]),
                       ("rpc", report["ml_and_rpc"]["rpc_rows"])]:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (OUTPUT/(name+".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    electronic = report["prescribed_path_electronic_convergence"]["rows"]
    steps = np.array([r["dt"] for r in electronic])
    error = np.array([r["coefficient_error"] for r in electronic])
    axes[0].loglog(steps, error, "o-", color="#2471a3", label="Overlap propagation")
    axes[0].loglog(steps, error[-1]*(steps/steps[-1])**2, "--", color="0.4", label="Second-order guide")
    axes[0].set(title="Smooth prescribed electronic path", xlabel="Timestep (atomic units)",
                ylabel="Electronic coefficient error")
    events = report["mash_exact_event_comparison"]["rows"]
    steps = np.array([r["dt"] for r in events])
    error = np.array([r["position_error"] for r in events])
    axes[1].loglog(steps, error, "o-", color="#ba4a00", label="Endpoint MASH")
    axes[1].loglog(steps, [r["exact_endpoint_delay_prediction"] for r in events], "--", color="0.4",
                   label="Exact half-step delay error")
    axes[1].set(title="Single analytically located MASH hop", xlabel="Timestep (atomic units)",
                ylabel="Nuclear coordinate error (bohr)")
    axes[1].text(.05, .85, "Observed order: 1.0000\nEnergy error: 0.0 (reported precision)",
                 transform=axes[1].transAxes, fontsize=9)
    for ax in axes:
        ax.grid(True, which="both", alpha=.2)
        ax.legend(loc="lower right", fontsize=8)
    fig.savefig(OUTPUT/"electronic_and_mash_convergence.png", dpi=180)
    fig.savefig(OUTPUT/"electronic_and_mash_convergence.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
