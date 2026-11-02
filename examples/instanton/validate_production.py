"""Reproduce T09 instanton accuracy and actual PES-call measurements.

Run from a source checkout with a compiled PyRPMD installation::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/instanton/validate_production.py \
        --baseline-ref 8cc402f
    python benchmarks/h3/build_pes.py
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/instanton/validate_production.py \
        --h3 --h3-beads 32 64 128 256 --output instanton-validation.json
    # At 200 K the coarse P=32/64 phase mode is unresolved; use an explicit finer grid.
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/instanton/validate_production.py \
        --h3 --temperatures 200 --h3-beads 128 256 384 --output instanton-h3-200.json

An Eckart P scan approaches the continuum *semiclassical* answer. Its difference
from the exact quantum transmission integral is reported independently. The
optional H3 calculation uses the genuine checksum-verified BKMP2 library and
all nine Cartesian coordinates, with gas-phase translational/rotational factors.
No fitted surrogate or stored literature rate substitutes for that calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import types

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

# The optional benchmark provider belongs to the source checkout.
REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from pyrpmd.constants import AMU_TO_AU, KB_AU
from pyrpmd.instanton import instanton_rate_1d, instanton_seed, optimize_instanton
from pyrpmd.low_temperature import HarmonicState
from pyrpmd.potentials.base import PESResult, coordinates


REFERENCES = {
    "h3_rates": {
        "doi": "10.1063/1.4996339",
        "citation": "Karandashev et al., Structural Dynamics 4, 061501 (2017), Table III",
        "url": "https://www.nccr-must.ch/libraries.files/karandashev-20171.pdf",
        "potential": "BKMP2",
        "crossover_temperature_kelvin": 345.0,
        "rate_unit": "cm^3/s",
        "semiclassical_rates": {300: 5.2e-16, 250: 4.3e-17, 200: 2.2e-18},
        "exact_quantum_rates": {300: 3.24e-16, 250: 3.67e-17, 200: 2.01e-18},
        "precision": "two significant figures; up to 512 beads in the source",
    },
    "degeneracy": {
        "doi": "10.1080/0144235X.2018.1472353",
        "citation": "Richardson, Int. Rev. Phys. Chem. 37, 171-216 (2018), Table II and preceding paragraph",
        "convention": "distinguishable rotors (sigma=1); multiply by two equivalent H2 attack channels",
    },
    "prefactor": "10.1021/acs.jpclett.6b02115",
    "exact_eckart_transmission": "10.1103/PhysRev.35.1303",
}


class EckartPotential:
    """V0 sech^2(x/a), with optional separable harmonic Cartesian spectators."""

    def __init__(self, height=0.01, width_bohr=2.0, bath_curvature=0.01):
        self.height = float(height)
        self.width = float(width_bohr)
        self.bath = float(bath_curvature)

    def evaluate(self, path):
        q = coordinates(path)
        t = np.tanh(q[:, 0, 0] / self.width)
        force = -self.bath * q
        force[:, 0, 0] = 2 * self.height / self.width * t * (1 - t*t)
        spectator = q.copy()
        spectator[:, 0, 0] = 0
        energy = self.height * (1-t*t) + 0.5*self.bath*np.sum(spectator**2, axis=(1, 2))
        return PESResult(energy, force)


class AnalyticEckartPotential(EckartPotential):
    """Same PES with an analytic unweighted Cartesian Hessian provider."""

    def hessian(self, path):
        q = coordinates(path)
        size = q.shape[1]*3
        out = np.broadcast_to(self.bath*np.eye(size), (len(q), size, size)).copy()
        t = np.tanh(q[:, 0, 0] / self.width)
        out[:, 0, 0] = 2*self.height/self.width**2*(1-t*t)*(3*t*t-1)
        return out


class CountedPotential:
    """Observe provider calls and evaluated frames without changing the PES."""

    def __init__(self, potential):
        self.potential = potential
        self.force_calls = 0
        self.force_frames = 0
        self.hessian_calls = 0
        self.hessian_frames = 0

    def evaluate(self, q):
        self.force_calls += 1
        self.force_frames += len(q)
        return self.potential.evaluate(q)

    def __getattr__(self, name):
        if name != "hessian":
            return getattr(self.potential, name)
        provider = getattr(self.potential, name)  # Absent stays absent.

        def counted(q):
            self.hessian_calls += 1
            self.hessian_frames += len(q)
            return provider(q)

        return counted

    def counters(self):
        return {key: getattr(self, key) for key in
                ("force_calls", "force_frames", "hessian_calls", "hessian_frames")}


def _log_sinh_positive(value):
    if value <= 0:
        return -math.inf
    return value + math.log(-math.expm1(-2*value)) - math.log(2)


def eckart_reference(height, width_bohr, mass, temperature_kelvin):
    """Independent closed continuum SCI and exact sech-squared scattering.

    The exact transmission used here is the textbook symmetric-barrier solution
    sinh^2(pi*a*sqrt(2*m*E)) / [sinh^2(...) + cosh^2(pi*sqrt(2*m*a*a*V0-1/4))].
    For shallow barriers the cosh becomes cos. No production Eckart helper is
    called. Integrating in beta*E and scaling around the maximum resolves tiny
    fluxes without relying on a permissive absolute quadrature tolerance.
    """
    values = (height, width_bohr, mass, temperature_kelvin)
    if any(not math.isfinite(float(v)) or float(v) <= 0 for v in values):
        raise ValueError("Eckart parameters must be positive and finite")
    beta = 1/(KB_AU*temperature_kelvin)
    omega = math.sqrt(2*height/mass)/width_bohr
    beta_c = 2*math.pi/omega
    if beta <= beta_c:
        raise ValueError("continuum thermal instanton requires temperature below crossover")
    alpha = beta_c*height
    stationary_energy = height*(beta_c/beta)**2
    action = alpha*(2-beta_c/beta)
    log_sci = 0.5*math.log(stationary_energy/(math.pi*beta))-action
    discriminant = 2*mass*width_bohr**2*height-0.25
    if discriminant >= 0:
        z = math.pi*math.sqrt(discriminant)
        log_barrier = 2*(z+math.log1p(math.exp(-2*z))-math.log(2))
    else:
        log_barrier = 2*math.log(abs(math.cos(math.pi*math.sqrt(-discriminant))))

    def log_weight(u):
        x = math.pi*width_bohr*math.sqrt(2*mass*u/beta)
        log_numerator = 2*_log_sinh_positive(x)
        log_probability = log_numerator-float(np.logaddexp(log_numerator, log_barrier))
        return log_probability-u

    upper = max(beta*height+60, 80.0)
    maximum = minimize_scalar(lambda u: -log_weight(u), bounds=(1e-15, upper), method="bounded")
    peak = log_weight(float(maximum.x))
    integral, error = quad(lambda u: math.exp(log_weight(u)-peak), 0, upper,
                           points=[float(maximum.x), beta*height], epsabs=1e-11,
                           epsrel=1e-11, limit=300)
    log_exact = math.log(integral)+peak-math.log(2*math.pi*beta)
    log_classical = -beta*height-math.log(2*math.pi*beta)
    return {
        "beta_au": beta, "crossover_temperature_kelvin": 1/(KB_AU*beta_c),
        "continuum_action_hbar": action,
        "continuum_energy_hartree": stationary_energy,
        "continuum_spring_action_hbar": alpha*(1-beta_c/beta),
        "continuum_sci_flux_au": math.exp(log_sci),
        "log_continuum_sci_flux_au": log_sci,
        "exact_quantum_flux_au": math.exp(log_exact),
        "log_exact_quantum_flux_au": log_exact,
        "continuum_sci_over_exact": math.exp(log_sci-log_exact),
        "continuum_sci_tunnelling_factor": math.exp(log_sci-log_classical),
        "exact_quantum_tunnelling_factor": math.exp(log_exact-log_classical),
        "quadrature_relative_error_estimate": error/integral,
        "omitted_tail_relative_upper_bound": math.exp(-upper-math.log(2*math.pi*beta)-log_exact),
        "interpretation": "P convergence resolves discretization; SCI versus exact is a distinct approximation error",
    }


def eckart_scan(bead_counts=(32, 64, 128), *, temperature_ratio=0.5,
                gradient_tolerance=1e-8):
    potential, mass = EckartPotential(), 1822.0
    omega = math.sqrt(2*potential.height/mass)/potential.width
    temperature = temperature_ratio*omega/(2*math.pi*KB_AU)
    reference = eckart_reference(potential.height, potential.width, mass, temperature)
    active = np.array([[True, False, False]])
    rows = []
    amplitude = math.sqrt(potential.height/reference["continuum_energy_hartree"]-1)
    for beads in bead_counts:
        # Exact continuum orbit sampled at the link-reflection grid.
        seed = np.zeros((beads, 1, 3))
        seed[:, 0, 0] = potential.width*np.arcsinh(
            amplitude*np.cos(2*math.pi*(np.arange(beads)+0.5)/beads))
        counted = CountedPotential(potential)
        started = time.perf_counter()
        rate = instanton_rate_1d(counted, [mass], temperature, seed, active_mask=active,
                                reactant_partition=1.0, gradient_tolerance=gradient_tolerance,
                                method="bofill", max_iterations=500)
        log_flux = rate["log_rate_au"]
        orbit = rate["instanton"]
        rows.append({
            "nbeads": int(beads), "action_hbar": orbit["action_hbar"],
            "gradient_max": orbit["gradient_max"],
            "flux_au": math.exp(log_flux), "log_flux_au": log_flux,
            "relative_error_vs_continuum_sci": math.expm1(log_flux-reference["log_continuum_sci_flux_au"]),
            "relative_error_vs_exact_quantum": math.expm1(log_flux-reference["log_exact_quantum_flux_au"]),
            "elapsed_seconds": time.perf_counter()-started,
            "beta_derivative_relative_difference": rate["beta_derivative_relative_difference"],
            **counted.counters(),
        })
    return {"temperature_kelvin": temperature, "temperature_over_crossover": temperature_ratio,
            "parameters": {"height_hartree": potential.height, "width_bohr": potential.width,
                           "mass_electron": mass}, "reference": reference, "rows": rows,
            "physical_converged": False}


def _legacy_optimizer(reference):
    """Load an explicitly requested immutable source revision for measurement."""
    commit = subprocess.run(["git", "rev-parse", "--verify", f"{reference}^{{commit}}"],
                            cwd=REPOSITORY, check=True, capture_output=True, text=True).stdout.strip()
    source = subprocess.run(["git", "show", f"{commit}:src/pyrpmd/instanton.py"],
                            cwd=REPOSITORY, check=True, capture_output=True, text=True).stdout
    name = f"pyrpmd._validation_legacy_instanton_{commit}"
    module = types.ModuleType(name)
    module.__package__ = "pyrpmd"
    sys.modules[name] = module  # dataclasses resolves the defining module here.
    exec(compile(source, f"{commit}:src/pyrpmd/instanton.py", "exec"), module.__dict__)
    return module.optimize_instanton, {"commit": commit,
                                       "source_sha256": hashlib.sha256(source.encode()).hexdigest()}


def optimizer_benchmark(*, nbeads=32, natoms=6, gradient_tolerance=1e-8, repeats=3,
                        baseline_ref=None):
    """Same PES, seed, tolerance, final Hessian and thread environment for both."""
    mass = np.full(natoms, 1822.0)
    potential = EckartPotential()
    temperature = 0.5*math.sqrt(2*potential.height/mass[0])/potential.width/(2*math.pi*KB_AU)
    ts = np.zeros((natoms, 3))
    direction = ts.copy()
    direction[0, 0] = 1
    seed = instanton_seed(ts, direction, nbeads, 2.5)
    rows = []
    for hessian_kind, cls in (("finite_difference", EckartPotential), ("analytic_provider", AnalyticEckartPotential)):
        for method in ("root", "bofill"):
            elapsed = []
            counts = []
            for _ in range(repeats):
                counted = CountedPotential(cls())
                started = time.perf_counter()
                orbit = optimize_instanton(counted, mass, temperature, seed, method=method,
                                           gradient_tolerance=gradient_tolerance,
                                           max_iterations=500, retain_hessian=True)
                elapsed.append(time.perf_counter()-started)
                counts.append(counted.counters())
            if any(count != counts[0] for count in counts):
                raise RuntimeError("non-reproducible deterministic PES counters")
            rows.append({"method": method, "hessian_source": hessian_kind,
                         "median_seconds": float(np.median(elapsed)), "elapsed_seconds": elapsed,
                         "gradient_max": orbit.gradient_max, "action_hbar": orbit.action_hbar,
                         "converged": orbit.converged, **counts[0]})
    comparisons = []
    for old, new in (rows[:2], rows[2:]):
        comparisons.append({"hessian_source": old["hessian_source"],
                            "root_over_bofill_force_calls": old["force_calls"]/new["force_calls"],
                            "root_over_bofill_evaluated_frames": old["force_frames"]/new["force_frames"],
                            "root_over_bofill_median_time": old["median_seconds"]/new["median_seconds"],
                            "absolute_action_difference_hbar": abs(old["action_hbar"]-new["action_hbar"])})
    baseline = None
    if baseline_ref is not None:
        legacy, provenance = _legacy_optimizer(baseline_ref)
        elapsed, counts = [], []
        for _ in range(repeats):
            counted = CountedPotential(EckartPotential())
            started = time.perf_counter()
            orbit = legacy(counted, mass, temperature, seed, gradient_tolerance=gradient_tolerance,
                           max_iterations=500, retain_hessian=True)
            elapsed.append(time.perf_counter()-started)
            counts.append(counted.counters())
        if any(count != counts[0] for count in counts):
            raise RuntimeError("non-reproducible legacy PES counters")
        baseline = {"method": "legacy_root", "hessian_source": "finite_difference",
                    "provenance": provenance, "elapsed_seconds": elapsed,
                    "median_seconds": float(np.median(elapsed)), "gradient_max": orbit.gradient_max,
                    "action_hbar": orbit.action_hbar, "converged": orbit.converged, **counts[0]}
        for target in (rows[1], rows[3]):
            comparisons.append({"baseline": "legacy_root_finite_difference",
                                "target": f"bofill_{target['hessian_source']}",
                                "root_over_bofill_force_calls": baseline["force_calls"]/target["force_calls"],
                                "root_over_bofill_evaluated_frames": baseline["force_frames"]/target["force_frames"],
                                "root_over_bofill_median_time": baseline["median_seconds"]/target["median_seconds"],
                                "absolute_action_difference_hbar": abs(baseline["action_hbar"]-target["action_hbar"])})
    return {"nbeads": nbeads, "natoms": natoms, "active_cartesian_dimensions": 3*natoms,
            "gradient_tolerance": gradient_tolerance, "repeats": repeats,
            "counting_scope": "all optimize_instanton PES calls, including initial and final Hessians",
            "rows": rows, "comparisons": comparisons, "legacy_baseline": baseline}


class IsolatedBKMP2Hydrogen:
    """Rotationally invariant H2 from the BKMP2 isolated-diatom radial limit."""

    def __init__(self, potential):
        self.potential = potential

    def evaluate(self, path):
        q = coordinates(path)
        if q.shape[1:] != (2, 3):
            raise ValueError("isolated H2 requires two atoms")
        vector = q[:, 1]-q[:, 0]
        radius = np.linalg.norm(vector, axis=1)
        if np.any(radius <= 0):
            raise ValueError("H2 bond length must be positive")
        data = [self.potential.radial([float(r), 99.0, 99.0]) for r in radius]
        energy = np.array([row[0] for row in data])
        gradient = np.array([row[1][0] for row in data])
        force = np.zeros_like(q)
        force[:, 1] = -gradient[:, None]*vector/radius[:, None]
        force[:, 0] = -force[:, 1]
        return PESResult(energy, force)


def h3_system(library=None):
    """Build stationary H3 TS and isolated H + H2 harmonic reactants."""
    from benchmarks.h3.pes import BKMP2Potential
    potential = BKMP2Potential(library)
    mass = 1.00782503223*AMU_TO_AU
    bond = brentq(lambda r: potential.radial([r, 99., 99.])[1][0], 1.2, 1.6, xtol=1e-13)
    ts_bond = brentq(lambda r: float(potential.radial([r, r, 2*r])[1] @ [1., 1., 2.]),
                     1.5, 2.0, xtol=1e-13)
    ts_geometry = np.array([[-ts_bond, 0., 0.], [0., 0., 0.], [ts_bond, 0., 0.]])
    ts = HarmonicState.from_potential(potential, ts_geometry, np.full(3, mass),
                                      rigid_body=True, hessian_step_bohr=1e-4,
                                      force_tolerance=1e-8, electronic_degeneracy=2)
    _, imaginary, _ = ts.spectrum(negative_modes=1)
    h2_geometry = np.array([[-bond/2, 0., 0.], [bond/2, 0., 0.]])
    h2 = HarmonicState.from_potential(IsolatedBKMP2Hydrogen(potential), h2_geometry,
                                     [mass, mass], rigid_body=True, symmetry_number=1,
                                     hessian_step_bohr=1e-4, force_tolerance=1e-8)
    atom = HarmonicState(np.zeros((1, 3)), [mass], 0.0, np.zeros((3, 3)),
                         rigid_body=True, electronic_degeneracy=2)
    _, vectors = np.linalg.eigh(ts.hessian/mass)
    direction = vectors[:, 0].reshape(3, 3)/math.sqrt(mass)
    diagnostics = {
        "h2_bond_bohr": bond, "ts_bond_bohr": ts_bond,
        "h2_energy_hartree": h2.energy_hartree, "ts_energy_hartree": ts.energy_hartree,
        "barrier_hartree": ts.energy_hartree-h2.energy_hartree,
        "crossover_temperature_kelvin": imaginary/(2*math.pi*KB_AU),
        "hydrogen_mass_amu": 1.00782503223, "active_cartesian_dimensions": 9,
        "reactant_rotor_symmetry_number": 1, "instanton_rotor_symmetry_number": 1,
        "reaction_degeneracy": 2, "reactant_h_doublet_degeneracy": 2,
        "instanton_doublet_degeneracy": 2,
        "energy_zero": "BKMP2 separated hydrogen atoms; action shifted by the isolated H2 minimum",
        "potential_identity": dict(potential.metadata),
    }
    return potential, np.full(3, mass), ts, [atom, h2], direction, diagnostics


def h3_scan(library=None, *, temperatures=(250., 300.), bead_counts=(32, 64, 128),
            gradient_tolerance=1e-8):
    from pyrpmd.workflows.instanton import run_instanton
    potential, masses, ts, reactants, direction, stationary = h3_system(library)
    reports = []
    for temperature in temperatures:
        if temperature >= stationary["crossover_temperature_kelvin"]:
            raise ValueError("H3 validation temperatures must be below crossover")
        seed = instanton_seed(ts.geometry_bohr, direction, bead_counts[0], 0.45)
        counted = CountedPotential(potential)
        started = time.perf_counter()
        report = run_instanton(
            counted, masses, seed, mode="rate", bead_counts=bead_counts,
            potential_identity=stationary["potential_identity"], temperature_kelvin=temperature,
            transition_state=ts.geometry_bohr, reactants=reactants,
            energy_reference_hartree=stationary["h2_energy_hartree"],
            optimizer_options={"method": "bofill", "gradient_tolerance": gradient_tolerance,
                               "max_iterations": 500, "phase_tolerance": 0.05},
            rate_options={"rigid_body": True, "symmetry_number": 1,
                          "reaction_degeneracy": 2, "electronic_degeneracy": 2,
                          "gradient_tolerance": gradient_tolerance, "phase_tolerance": 0.05},
            convergence_relative_tolerance=0.01,
        )
        sci_reference = REFERENCES["h3_rates"]["semiclassical_rates"].get(temperature)
        qm_reference = REFERENCES["h3_rates"]["exact_quantum_rates"].get(temperature)
        rows = []
        for row in report["rows"]:
            rate, orbit = row["rate"], row["orbit"]
            value = rate["rate"]*1e6
            rows.append({
                "nbeads": row["nbeads"], "rate_cm3_s": value,
                "relative_error_vs_literature_sci": None if sci_reference is None else value/sci_reference-1,
                "relative_error_vs_exact_quantum": None if qm_reference is None else value/qm_reference-1,
                "action_hbar": orbit["action_hbar"], "gradient_max": orbit["gradient_max"],
                "optimizer_diagnostics": orbit["optimizer_diagnostics"],
                "removed_rigid_modes": rate["removed_rigid_modes"],
                "phase_mode_relative_residual": rate["phase_mode_relative_residual"],
                "log_reactant_partition": rate["log_reactant_partition"],
                "log_prefactor_au": rate["log_prefactor_au"],
            })
        reports.append({"temperature_kelvin": temperature, "rows": rows,
                        "physics_fingerprint": report["physics_fingerprint"],
                        "bead_convergence": report["bead_convergence"],
                        "numerical_converged": report["numerical_converged"],
                        "discretization_converged": report["discretization_converged"],
                        "physical_converged": False,
                        "elapsed_seconds": time.perf_counter()-started, **counted.counters()})
    return {"stationary_points": stationary, "literature": REFERENCES["h3_rates"],
            "symmetry_convention": REFERENCES["degeneracy"], "temperatures": reports,
            "gradient_tolerance": gradient_tolerance, "bead_relative_tolerance": 0.01,
            "phase_tolerance": 0.05,
            "limitations": ["semiclassical and harmonic-reactant approximations",
                            "classical rigid rotors, no resolved nuclear-spin populations",
                            "bead convergence is not proof of exact quantum accuracy",
                            "dense prefactor supports at most 4096 active bead coordinates"]}


def hardware_metadata():
    processor = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                processor = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    import scipy
    return {"platform": platform.platform(), "processor": processor,
            "logical_cpu_count": os.cpu_count(), "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "threads": {key: os.environ.get(key) for key in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "validation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _atomic_json(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h3", action="store_true", help="Run genuine BKMP2 molecular rate validation")
    parser.add_argument("--library", type=Path, help="Prebuilt checksum-verified BKMP2 library")
    parser.add_argument("--output", type=Path, help="Atomically write the full JSON report")
    parser.add_argument("--beads", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--h3-beads", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[250., 300.])
    parser.add_argument("--gradient-tolerance", type=float, default=1e-8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline-ref", help="Also execute an explicit Git source revision (e.g. 8cc402f) for an actual legacy comparison")
    args = parser.parse_args(argv)
    if args.library is not None and not args.h3:
        parser.error("--library requires --h3")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    for beads in (args.beads, args.h3_beads):
        if len(beads) < 3 or any(p < 8 or p % 2 for p in beads) or any(b <= a for a, b in zip(beads, beads[1:])):
            parser.error("bead scans require at least three strictly increasing even counts >= 8")
    report = {"schema": "pyrpmd.instanton.production-validation.v1", "hardware": hardware_metadata(),
              "references": REFERENCES, "physical_converged": False}
    report["eckart"] = eckart_scan(args.beads, gradient_tolerance=args.gradient_tolerance)
    report["optimizer"] = optimizer_benchmark(gradient_tolerance=args.gradient_tolerance,
                                               repeats=args.repeats, baseline_ref=args.baseline_ref)
    if args.h3:
        report["h3"] = h3_scan(args.library, temperatures=args.temperatures, bead_counts=args.h3_beads,
                                gradient_tolerance=args.gradient_tolerance)
    else:
        report["h3"] = {"executed": False, "reason": "enable --h3 after building benchmarks/h3/build_pes.py"}
    if args.output is not None:
        _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
