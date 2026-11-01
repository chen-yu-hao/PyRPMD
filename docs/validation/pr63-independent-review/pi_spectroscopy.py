"""Independent nonlinear action and spectroscopy review for PR #63.

Run from the repository root:
  OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python \
    docs/validation/pr63-independent-review/pi_spectroscopy.py

Only NumPy/SciPy and production PyRPMD are needed. The nonharmonic oscillator
is Cao--Voth, JCP 101, 6157 (1994), Eq. (5.1): m=hbar=1,
V(x)=x**2/2+0.10*x**3+0.01*x**4. It is a new method-level comparison,
not a reproduction of the paper's real-time momentum-correlation figure.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.special import logsumexp

from pyrpmd.analysis.spectroscopy import relative_raman_spectrum, relative_sfg_spectrum
from pyrpmd.cmd import PACMDDynamics
from pyrpmd.dynamics import State
from pyrpmd.pimd_methods import (GaussianTrialWavefunction, PIGSPotential,
                                SuzukiChinPotential, bosonic_exchange)
from pyrpmd.potentials.base import PESResult

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
UPSTREAM = "3faac3aa42b5424b2711367608e0138c82f62e0a"
SEED = 631105


def v(x):
    return 0.5*x*x + 0.10*x**3 + 0.01*x**4


def dv(x):
    return x + 0.30*x*x + 0.04*x**3


class Oscillator:
    """A one-dimensional oracle embedded in the production [P,1,3] contract."""
    def evaluate(self, q):
        f = np.zeros_like(q)
        f[:, 0, 0] = -dv(q[:, 0, 0])
        return PESResult(v(q[:, 0, 0]), f)


def quantum_reference(size, beta=10.0):
    # Extra basis states ensure q**4 is not formed with a truncated boundary.
    q = np.diag(np.sqrt(np.arange(1, size+8)/2), 1)
    q += q.T
    q2, q3 = q@q, q@q@q
    q4 = q2@q2
    potential = (0.5*q2 + .1*q3 + .01*q4)[:size, :size]
    h = np.diag(np.arange(size)+.5) + (.1*q3+.01*q4)[:size, :size]
    energies, vectors = eigh(h)
    weights = np.exp(-beta*(energies-energies[0]))
    weights /= weights.sum()
    vm = vectors.T@potential@vectors
    qm = vectors.T@q[:size, :size]@vectors
    q2m = vectors.T@q2[:size, :size]@vectors
    summary = {"basis_size": size, "ground_energy": float(energies[0]),
               "fundamental": float(energies[1]-energies[0]),
               "thermal_potential": float(weights@np.diag(vm)),
               "thermal_x": float(weights@np.diag(qm)),
               "thermal_x2": float(weights@np.diag(q2m)),
               "ground_potential": float(vm[0, 0]), "ground_x2": float(q2m[0, 0])}
    return summary, (energies, vectors, potential, q2[:size, :size])


def coordinate_grid_reference():
    """An independent coordinate-space Schrödinger check of the basis oracle."""
    rows = []
    for dx in (.008, .004, .002):
        x = np.arange(-12.+dx, 12.-dx/2, dx)
        energies = eigh_tridiagonal(1/dx**2+v(x), np.full(len(x)-1, -.5/dx**2),
                                   select='i', select_range=(0, 1), eigvals_only=True)
        rows.append({"dx": dx, "ground_energy": float(energies[0]),
                     "fundamental": float(energies[1]-energies[0])})
    extrapolated = {key: (4*rows[-1][key]-rows[-2][key])/3
                    for key in ("ground_energy", "fundamental")}
    return {"rows": rows, "richardson_extrapolated": extrapolated}


def density_power(kernel, exponent):
    eig, vec = eigh(kernel)
    # Tiny negative eigenvalues of a positive kernel arise only from roundoff.
    if eig.min() < -1e-12*eig.max():
        raise AssertionError("transfer kernel lost positive semidefiniteness")
    weights = np.maximum(eig/eig[-1], 0.0)**exponent
    return vec**2@weights / weights.sum()


def sc_transfer(beads, dx, beta=10.0):
    x = np.linspace(-10., 10., round(20/dx)+1)
    dx = x[1]-x[0]
    tau = beta/beads
    g = dx*np.exp(-(x[:, None]-x[None, :])**2/(2*tau))/np.sqrt(2*np.pi*tau)
    potential = v(x)
    primitive = np.exp(-.5*tau*potential)
    primitive_density = density_power(primitive[:, None]*g*primitive[None, :], beads)
    model = SuzukiChinPotential(Oscillator(), [1.], beta, beads, fd_step=1e-5)
    even, odd = [], []
    # Read the actual production energies, without reusing its coefficients.
    q = np.zeros((beads, 1, 3))
    for point in x:
        q[:, 0, 0] = point
        energy = model.evaluate(q).energy
        even.append(energy[0])
        odd.append(energy[1])
    de = np.exp(-.5*tau*np.array(even))
    do = np.exp(-tau*np.array(odd))
    kernel = de[:, None]*((g*do[None, :])@g)*de[None, :]
    density = density_power(kernel, beads//2)
    return {"beads": beads, "grid_dx": float(dx),
            "sc_potential": float(density@potential),
            "primitive_potential": float(primitive_density@potential),
            "sc_x": float(density@x), "sc_x2": float(density@(x*x))}


def sc_comparison(reference):
    rows = [sc_transfer(p, .05) for p in (8, 16, 32, 64, 128)]
    exact = reference["thermal_potential"]
    for row in rows:
        row["sc_absolute_error"] = abs(row["sc_potential"]-exact)
        row["primitive_absolute_error"] = abs(row["primitive_potential"]-exact)
    for before, after in zip(rows, rows[1:]):
        after["sc_order"] = math.log2(before["sc_absolute_error"]/after["sc_absolute_error"])
        after["primitive_order"] = math.log2(before["primitive_absolute_error"]/after["primitive_absolute_error"])
    fine = sc_transfer(128, .025)
    assert rows[-1]["sc_order"] > 3.9
    assert rows[-1]["sc_absolute_error"] < rows[-1]["primitive_absolute_error"]/100
    return {"beta": 10.0, "rows": rows,
            "grid_check": {"coarse_dx": .05, "fine_dx": .025,
                           "sc_potential_difference": abs(rows[-1]["sc_potential"]-fine["sc_potential"]),
                           "primitive_potential_difference": abs(rows[-1]["primitive_potential"]-fine["primitive_potential"])}}


class CoupledPotential:
    def __init__(self):
        rng = np.random.default_rng(SEED)
        a = rng.normal(size=(6, 6))*.2
        self.k = a.T@a+np.diag(np.linspace(.3, 1.1, 6))

    def evaluate(self, q):
        z = q.reshape(len(q), -1)
        energy = .5*np.einsum('bi,ij,bj->b', z, self.k, z)+.03*np.sum(z**4, axis=1)
        energy += .07*z[:, 0]**2*z[:, 3]**2
        gradient = z@self.k+.12*z**3
        gradient[:, 0] += .14*z[:, 0]*z[:, 3]**2
        gradient[:, 3] += .14*z[:, 3]*z[:, 0]**2
        return PESResult(energy, -gradient.reshape(q.shape))

    def hessian(self, q):
        z = q.reshape(-1)
        h = self.k+np.diag(.36*z*z)
        h[0, 0] += .14*z[3]**2
        h[3, 3] += .14*z[0]**2
        h[0, 3] += .28*z[0]*z[3]
        h[3, 0] += .28*z[0]*z[3]
        return h


def coupled_sc_force():
    rng = np.random.default_rng(SEED+1)
    q = rng.normal(scale=.6, size=(8, 2, 3))
    masses, beta = np.array([1.3, 2.7]), 4.5
    physical = CoupledPotential()
    f = physical.evaluate(q).forces
    rows = []
    for alpha in (0., .37, 1.):
        for step in (1e-2, 1e-3, 1e-4, 1e-5):
            model = SuzukiChinPotential(physical, masses, beta, len(q), alpha=alpha, fd_step=step)
            # Oracle comes directly from differentiating the published Eq. (7).
            weights = np.where(np.arange(len(q))%2 == 0, 2/3, 4/3)
            ds = np.where(np.arange(len(q))%2 == 0, alpha/6, (1-alpha)/12)
            exact = np.empty_like(q)
            for j in range(len(q)):
                projected = physical.hessian(q[j])@(f[j]/masses[:, None]).reshape(-1)
                exact[j] = weights[j]*(f[j]+2*ds[j]*(beta/len(q))**2*projected.reshape(2, 3))
            error = np.max(np.abs(model.evaluate(q).forces-exact))
            rows.append({"alpha": alpha, "fd_step": step, "max_force_error": float(error)})
    assert max(row["max_force_error"] for row in rows if row["fd_step"] == 1e-5) < 1e-8
    return rows


def cubic_boson_oracle(q, mass, beta):
    p, n, _ = q.shape
    coefficient, beta_p = mass*(p/beta)**2, beta/p
    energy, force = [0.], [np.zeros_like(q)]
    for stop in range(1, n+1):
        cycle_energy, cycle_force = [], []
        for start in range(stop):
            nodes = [(b, a) for a in range(start, stop) for b in range(p)]
            e, f = 0., np.zeros_like(q)
            for a, b in zip(nodes, nodes[1:]+nodes[:1]):
                delta = q[a]-q[b]
                e += .5*coefficient*delta@delta
                f[a] -= coefficient*delta
                f[b] += coefficient*delta
            cycle_energy.append(energy[start]+e)
            cycle_force.append(force[start]+f)
        logits = -beta_p*np.array(cycle_energy)
        normalizer = logsumexp(logits)
        energy.append(-(normalizer-np.log(stop))/beta_p)
        force.append(np.einsum('i,ipnc->pnc', np.exp(logits-normalizer), cycle_force))
    return energy[-1], force[-1]


def boson_comparison():
    rng = np.random.default_rng(SEED+2)
    rows = []
    for p in (1, 2, 7):
        for n in (1, 2, 5, 9):
            for beta in (.2, 3., 20.):
                q = rng.normal(size=(p, n, 3))
                expected_e, expected_f = cubic_boson_oracle(q, 1.7, beta)
                result = bosonic_exchange(q, 1.7, beta)
                row = {"beads": p, "particles": n, "beta": beta,
                       "relative_energy_error": abs(result.energy-expected_e)/max(1., abs(expected_e)),
                       "relative_max_force_error": float(np.max(np.abs(result.forces-expected_f))/max(1., np.max(np.abs(expected_f)))),
                       "row_sum_error": float(np.max(abs(result.connections.sum(axis=1)-1))),
                       "column_sum_error": float(np.max(abs(result.connections.sum(axis=0)-1)))}
                rows.append(row)
    assert max(r["relative_max_force_error"] for r in rows) < 2e-10
    return {"cases": len(rows), "rows": rows}


def ho_functions(x, size):
    phi = np.empty((len(x), size))
    phi[:, 0] = np.pi**(-.25)*np.exp(-x*x/2)
    phi[:, 1] = np.sqrt(2)*x*phi[:, 0]
    for j in range(1, size-1):
        phi[:, j+1] = np.sqrt(2/(j+1))*x*phi[:, j]-np.sqrt(j/(j+1))*phi[:, j-1]
    return phi


def pigs_comparison(qdata):
    energies, vectors, potential_matrix, x2_matrix = qdata
    x = np.linspace(-10., 10., 801)
    dx = x[1]-x[0]
    alpha = .25
    trial = np.exp(-.5*alpha*x*x)
    trial_basis = ho_functions(x, len(energies)).T@trial*dx
    trial_eigen = vectors.T@trial_basis
    vm = vectors.T@potential_matrix@vectors
    x2m = vectors.T@x2_matrix@vectors

    def exact_projection(duration):
        a = trial_eigen*np.exp(-.5*duration*(energies-energies[0]))
        a /= np.linalg.norm(a)
        return float(a@vm@a), float(a@x2m@a)

    kernels = {}
    def finite_projection(duration, links):
        tau = duration/links
        if tau not in kernels:
            g = dx*np.exp(-(x[:, None]-x[None, :])**2/(2*tau))/np.sqrt(2*np.pi*tau)
            d = np.exp(-tau*v(x)/2)
            kernels[tau] = eigh(d[:, None]*g*d[None, :])
        ev, vec = kernels[tau]
        a = vec@((vec.T@trial)*np.maximum(ev/ev[-1], 0.)**(links//2))
        probability = a*a/np.sum(a*a)
        return float(probability@v(x)), float(probability@(x*x))

    projection = []
    for duration in (1., 2., 4., 8., 16.):
        links = round(duration/.03125)
        vp, x2 = finite_projection(duration, links)
        exact_v, exact_x2 = exact_projection(duration)
        projection.append({"projection_time": duration, "links": links, "tau": .03125,
                           "potential": vp, "continuum_finite_projection_potential": exact_v,
                           "finite_tau_error": vp-exact_v,
                           "projection_error": exact_v-vm[0, 0],
                           "x2": x2, "continuum_finite_projection_x2": exact_x2})
    rows = []
    exact_v, _ = exact_projection(8.)
    for links in (16, 32, 64, 128, 256, 512):
        vp, x2 = finite_projection(8., links)
        row = {"projection_time": 8., "links": links, "tau": 8./links,
               "potential": vp, "absolute_error": abs(vp-exact_v), "x2": x2}
        if rows:
            row["order"] = math.log2(rows[-1]["absolute_error"]/row["absolute_error"])
        rows.append(row)
    # Check the sampled action independently, including endpoint half weights.
    action_checks = []
    rng = np.random.default_rng(SEED+3)
    for beads in (3, 9, 33):
        model = PIGSPotential(Oscillator(), [1.], beads, 4., GaussianTrialWavefunction(alpha))
        q = rng.normal(scale=.7, size=(beads, 1, 3))
        q[:, 0, 1:] = 0.
        points = q[:, 0, 0]
        tau = 4./(beads-1)
        oracle = (np.sum(np.diff(points)**2)/(2*tau)
                  +tau*(np.sum(v(points))-.5*(v(points[0])+v(points[-1])))
                  +.5*alpha*(points[0]**2+points[-1]**2))
        action_checks.append({"beads": beads, "absolute_action_error": abs(model.action(q)-oracle)})
    assert rows[-1]["order"] > 1.99
    assert max(a["absolute_action_error"] for a in action_checks) < 1e-10
    return {"trial_alpha": alpha, "projection_sweep": projection, "tau_sweep": rows,
            "action_checks": action_checks, "ground_potential": float(vm[0, 0]),
            "fixed_projection_exact_potential": exact_v}


def pacmd_comparison():
    rng = np.random.default_rng(SEED+4)
    a = rng.normal(size=(6, 6))*.3
    k = a.T@a+np.diag(np.linspace(.5, 1.3, 6))
    masses = np.array([1.4, 2.2])
    mass = np.repeat(masses, 3)
    class Harmonic:
        def evaluate(self, q):
            z = q.reshape(len(q), -1)
            return PESResult(.5*np.einsum('bi,ij,bj->b', z, k, z), -(z@k).reshape(q.shape))
    frequencies2, vectors = eigh(k/np.sqrt(mass[:, None]*mass[None, :]))
    frequencies = np.sqrt(frequencies2)
    q0, p0 = rng.normal(scale=.4, size=(2, 8, 2, 3))
    qc, pc = q0.mean(axis=0).ravel(), p0.mean(axis=0).ravel()
    y0, yp0 = vectors.T@(np.sqrt(mass)*qc), vectors.T@(pc/np.sqrt(mass))
    duration = 5.
    exact = vectors@(np.cos(frequencies*duration)*y0+np.sin(frequencies*duration)/frequencies*yp0)/np.sqrt(mass)
    rows = []
    for dt in (.04, .02, .01):
        dyn = PACMDDynamics(Harmonic(), masses, 3., 8, dt, target_frequency=8., seed=SEED+5)
        state = State(q0.copy(), p0.copy())
        for _ in range(round(duration/dt)):
            dyn.step(state)
        error = float(np.max(np.abs(state.q.mean(axis=0).ravel()-exact)))
        row = {"dt": dt, "duration": duration, "max_centroid_error": error}
        if rows:
            row["order"] = math.log2(rows[-1]["max_centroid_error"]/error)
        rows.append(row)
    assert rows[-1]["order"] > 1.98
    return {"masses": masses.tolist(), "coupled_frequencies": frequencies.tolist(), "rows": rows,
            "note": "Analytic harmonic centroid dynamics only; no anharmonic CMD or molecular-spectrum accuracy claim."}


def spectral_checks():
    dt, omega, beta = .02, 1.5, 2.
    t = np.arange(2049)*dt
    phases = np.arange(32)*2*np.pi/32
    wave = np.cos(omega*t[None, :]+phases[:, None])
    dipole = np.zeros((len(phases), len(t), 1, 3))
    dipole[:, :, 0, 2] = wave
    alpha = np.zeros((len(phases), len(t), 1, 3, 3))
    alpha[:, :, 0, 0, 0] = 1.3*wave
    alpha[:, :, 0, 1, 1] = .7*wave
    alpha[:, :, 0, 2, 2] = -.2*wave
    alpha[:, :, 0, 0, 1] = alpha[:, :, 0, 1, 0] = .4*wave
    spectrum = relative_raman_spectrum(alpha, dt, beta=beta, normalize=False)
    rng = np.random.default_rng(SEED+6)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = np.einsum('ij,...jk,lk->...il', rotation, alpha, rotation)
    rotated_spectrum = relative_raman_spectrum(rotated, dt, beta=beta, normalize=False)
    rotation_error = np.max(abs(spectrum["relative_raman"]-rotated_spectrum["relative_raman"]))/np.max(spectrum["relative_raman"])
    sfg = relative_sfg_spectrum(dipole, alpha, dt, beta=beta, max_lag=1000, demean=False, normalize=False)
    inverted = relative_sfg_spectrum(-dipole, alpha, dt, beta=beta, max_lag=1000, demean=False, normalize=False)
    lag = sfg["lag_time_au"]
    exact_corr = .65*np.cos(omega*lag)
    exact_response = beta*.65*omega*np.sin(omega*lag)
    # Exact continuous Fourier integral for the cosine-tapered sinusoidal response.
    duration = lag[-1]
    frequencies = sfg["angular_frequency_au"]
    def integral_exponential(a):
        return duration*np.sinc(a*duration/(2*np.pi))*np.exp(.5j*a*duration)
    def integral_sine(w):
        return (integral_exponential(frequencies+w)-integral_exponential(frequencies-w))/(2j)
    expected = beta*.65*omega*(.5*integral_sine(omega)
               +.25*(integral_sine(omega+np.pi/duration)+integral_sine(omega-np.pi/duration)))
    # Exclude near-Nyquist bins when quoting a low-frequency physical response error.
    support = frequencies <= 4*omega
    causal_error = np.linalg.norm(sfg["susceptibility"][support]-expected[support])/np.linalg.norm(expected[support])
    result = {"raman_global_rotation_relative_error": float(rotation_error),
              "sfg_phase_ensemble_correlation_max_error": float(np.max(abs(sfg["correlation"]-exact_corr))),
              "sfg_response_max_error": float(np.max(abs(sfg["response"]-exact_response))),
              "sfg_analytic_causal_response_relative_l2_error_omega_le_6": float(causal_error),
              "sfg_inversion_susceptibility_max_error": float(np.max(abs(inverted["susceptibility"]+sfg["susceptibility"]))),
              "sfg_inversion_intensity_max_error": float(np.max(abs(inverted["relative_sfg_intensity"]-sfg["relative_sfg_intensity"]))),
              "sample_interval": dt, "max_lag": 1000, "phase_replicas": len(phases)}
    assert rotation_error < 1e-12
    assert causal_error < 1e-3
    return result


def main():
    started = time.perf_counter()
    sources = ["src/pyrpmd/pimd_methods.py", "src/pyrpmd/cmd.py", "src/pyrpmd/analysis/spectroscopy.py",
               "docs/validation/pr63-independent-review/pi_spectroscopy.py"]
    hashes = {path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in sources}
    reference80, _ = quantum_reference(80)
    reference160, qdata = quantum_reference(160)
    grid_reference = coordinate_grid_reference()
    assert abs(grid_reference["richardson_extrapolated"]["fundamental"]-reference160["fundamental"]) < 2e-9
    results = {"upstream_commit": UPSTREAM, "seed": SEED, "source_sha256": hashes,
               "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                           "platform": platform.platform()},
               "model": {"potential": "x^2/2 + 0.10*x^3 + 0.01*x^4", "mass": 1., "hbar": 1.,
                         "source": "Cao and Voth, JCP 101, 6157 (1994), Eq. (5.1), doi:10.1063/1.468503"},
               "basis_convergence": [reference80, reference160],
               "coordinate_space_schrodinger_check": grid_reference,
               "paper_text_discrepancy": {
                   "printed_fundamental_redshift_percent": 6.,
                   "independently_calculated_redshift_percent": 100*(1-reference160["fundamental"]),
                   "interpretation": "The printed Eq. (5.1) parameters yield 4.06844%, not the adjacent prose's 6%; no parameter retuning or claim to reproduce that rounded sentence."}}
    for label, run in (("suzuki_chin", lambda: sc_comparison(reference160)),
                       ("sc_coupled_force", coupled_sc_force),
                       ("bosonic", boson_comparison),
                       ("pigs", lambda: pigs_comparison(qdata)),
                       ("pacmd", pacmd_comparison), ("spectral", spectral_checks)):
        results[label] = run()
        print(label+": complete", flush=True)
    results["wall_seconds"] = time.perf_counter()-started
    results["claims"] = {"method_level_checks_passed": True,
                         "published_molecular_system_reproduced": False,
                         "complete_tier2_method_coverage": False,
                         "sampling_convergence_certified": False}
    (OUT/"pi_spectroscopy_results.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps({"wall_seconds": results["wall_seconds"], "output": "pi_spectroscopy_results.json"}))


if __name__ == "__main__":
    main()
