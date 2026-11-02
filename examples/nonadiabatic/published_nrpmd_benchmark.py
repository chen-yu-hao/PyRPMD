"""Richardson--Thoss (2013), Fig. 1(c): signed thermal NRPMD correlations.

The published model is Eq. 15 with m=hbar=omega=kappa=beta=Delta=1 and
alpha=0.  The reference is an independent sinc-DVR spectral calculation of
the Kubo transform.  The example-specific ``Paper2013Dynamics`` uses the
paper's lowest-adiabatic-surface split with the existing PyRPMD MInt and
free-ring propagators.  The library's default trace split is also selectable.

No electrochemical reservoir or adaptive umbrella is used in this benchmark:
the paper validates the underlying nonadiabatic nuclear-quantum component.
Uncertainties retain signs and source-chain blocks, including their shared
normalization.  Sampling budgets are deliberately recorded rather than
claimed to reproduce the original paper's statistical precision.

Reference: J. O. Richardson and M. Thoss, J. Chem. Phys. 139, 031102 (2013),
doi:10.1063/1.4816124, Eqs. 7--15 and Fig. 1(c).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from pyrpmd.analysis.fes import statistical_inefficiency
from pyrpmd.nonadiabatic import DiabaticResult
from pyrpmd.potentials.base import coordinates
from pyrpmd.thermal_nrpmd import ThermalNRPMDDynamics, ThermalNRPMDSampler
import pyrpmd.thermal_nrpmd as thermal_module


PAPER = {
    "title": "Communication: Nonadiabatic ring-polymer molecular dynamics",
    "authors": "J. O. Richardson and M. Thoss",
    "journal": "J. Chem. Phys. 139, 031102 (2013)",
    "doi": "10.1063/1.4816124",
    "url": "https://doi.org/10.1063/1.4816124",
    "figure": "1(c)", "model_equation": 15, "correlation_equations": [12, 13, 14],
}


def _positive(value, name, *, zero=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    value = float(value)
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"invalid {name}")
    return value


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class PaperTwoStateModel:
    """Eq. 15, embedded in Cartesian xyz with separable harmonic y,z.

    The spectator oscillators cancel from normalized x/electronic quantities.
    They make the Cartesian sampler normalizable without fixed-coordinate
    machinery.  They exert no x or electronic force.
    """

    alpha: float = 0.0
    kappa: float = 1.0
    delta: float = 1.0

    def __post_init__(self):
        for name in ("alpha", "kappa", "delta"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and real")
            object.__setattr__(self, name, float(value))

    def evaluate(self, q):
        q = coordinates(q)
        if q.shape[1] != 1:
            raise ValueError("the paper model contains one Cartesian particle")
        x = q[:, 0, 0]
        common = 0.5*np.sum(q*q, axis=(1, 2))
        gap = self.alpha+self.kappa*x
        matrix = np.zeros((len(q), 2, 2))
        matrix[:, 0, 0], matrix[:, 1, 1] = common+gap, common-gap
        matrix[:, 0, 1] = matrix[:, 1, 0] = self.delta
        gradient = np.zeros((len(q), 2, 2, 1, 3))
        gradient[:, 0, 0] = q
        gradient[:, 1, 1] = q
        gradient[:, 0, 0, 0, 0] += self.kappa
        gradient[:, 1, 1, 0, 0] -= self.kappa
        return DiabaticResult(matrix, gradient)

    def lowest_adiabatic_split(self, q):
        full = self.evaluate(q)
        x = np.asarray(q)[:, 0, 0]
        gap = self.alpha+self.kappa*x
        radius = np.hypot(gap, self.delta)
        if np.any(radius == 0):
            raise ValueError("the lowest-adiabatic split is nonsmooth at an exact crossing")
        scalar = 0.5*np.sum(np.asarray(q)**2, axis=(1, 2))-radius
        scalar_gradient = np.array(q, dtype=float, copy=True)
        scalar_gradient[:, 0, 0] -= self.kappa*gap/radius
        matrix = full.matrix-scalar[:, None, None]*np.eye(2)
        gradients = full.gradients-scalar_gradient[:, None, None]*np.eye(2)[None, :, :, None, None]
        return SimpleNamespace(scalar=scalar, scalar_gradient=scalar_gradient,
                               mapping=DiabaticResult(matrix, gradients))


class Paper2013Dynamics(ThermalNRPMDDynamics):
    """Paper-specific partition using the library's unchanged exact MInt flow.

    The MInt kernel already includes the -Tr(V)/2 force.  Its corresponding
    energy correction must be restored because the parent class normally
    assumes a traceless mapping matrix.  This is an example adapter, not a
    change to the library's scalar/traceless default.
    """

    def __init__(self, potential, *args, **kwargs):
        if not isinstance(potential, PaperTwoStateModel):
            raise TypeError("Paper2013Dynamics requires PaperTwoStateModel")
        super().__init__(potential, *args, **kwargs)

    def _thermal_evaluate(self, q):
        return self.potential.lowest_adiabatic_split(q)

    def energy(self, state):
        energy = super().energy(state)
        mapping = self._thermal_evaluate(state.q).mapping.matrix
        return float(energy-0.5*np.trace(mapping, axis1=1, axis2=2).sum())


def exact_kubo(times, *, alpha=0.0, kappa=1.0, delta=1.0, beta=1.0,
               points=161, extent=9.0):
    """Independent sinc-DVR spectral Kubo autocorrelations of x and P_1.

    For eigenenergies E_a, the spectral weight is
    (exp(-beta E_a)-exp(-beta E_b))/(beta*(E_b-E_a)*Z), with its
    continuous limit on degenerate pairs.  Both observables are Hermitian,
    so summing cos((E_b-E_a)t)|A_ab|^2 gives their real Kubo transforms.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("times must be a finite nonempty vector")
    points = _integer(points, "points", 16)
    extent, beta = _positive(extent, "extent"), _positive(beta, "beta")
    model = PaperTwoStateModel(alpha, kappa, delta)
    grid = np.linspace(-extent, extent, points)
    spacing = float(grid[1]-grid[0])
    distance = np.arange(points)[:, None]-np.arange(points)[None, :]
    kinetic = np.empty((points, points))
    off_diagonal = distance != 0
    kinetic[off_diagonal] = (-1.0)**distance[off_diagonal]/(spacing*distance[off_diagonal])**2
    kinetic[~off_diagonal] = np.pi**2/(6*spacing**2)
    q = np.zeros((points, 1, 3))
    q[:, 0, 0] = grid
    matrix = model.evaluate(q).matrix
    hamiltonian = np.kron(kinetic, np.eye(2))
    for i in range(points):
        hamiltonian[2*i:2*i+2, 2*i:2*i+2] += matrix[i]
    energies, vectors = eigh(hamiltonian, check_finite=False)
    shifted = energies-energies[0]
    boltzmann = np.exp(-beta*shifted)
    partition = float(boltzmann.sum())
    gaps = energies[:, None]-energies[None, :]
    argument = beta*np.abs(gaps)
    factor = np.ones_like(argument)
    np.divide(-np.expm1(-argument), argument, out=factor, where=argument != 0)
    weights = np.maximum(boltzmann[:, None], boltzmann[None, :])*factor/partition
    operators = {"Cxx": np.repeat(grid, 2), "C11": np.tile([1., 0.], points)}
    curves = {}
    for name, diagonal in operators.items():
        representation = vectors.T @ (diagonal[:, None]*vectors)
        amplitude = weights*representation**2
        curves[name] = np.asarray([np.sum(amplitude*np.cos(gaps*t)) for t in times]).tolist()
    return {"method": "sinc_DVR_spectral_Kubo_transform", "time": times.tolist(), **curves,
            "points": points, "extent": extent, "spacing": spacing,
            "beta": beta, "alpha": model.alpha, "kappa": model.kappa, "delta": model.delta,
            "ground_energy": float(energies[0]),
            "log_partition_function_1d": float(np.log(partition)-beta*energies[0]),
            "mean_x": float(boltzmann @ np.diag(vectors.T @ (operators["Cxx"][:, None]*vectors))/partition),
            "population_1": float(boltzmann @ np.diag(vectors.T @ (operators["C11"][:, None]*vectors))/partition),
            "scope": "independently_recomputed_exact_quantum_reference_not_digitized_paper_data"}


@dataclass(frozen=True)
class BenchmarkConfig:
    beads: tuple = (1, 4, 8)
    chains: int = 4
    samples: int = 512
    burn_in: int = 1024
    stride: int = 4
    dt: float = 0.02
    duration: float = 5.0
    output_interval: float = 0.1
    bootstrap_replicates: int = 256
    block_multiplier: float = 2.0
    min_average_sign: float = 0.05
    refinement_parents: int = 8
    seed: int = 20261102
    potential_split: str = "lowest_adiabatic"

    def __post_init__(self):
        for name in ("chains", "samples", "stride", "bootstrap_replicates", "refinement_parents"):
            _integer(getattr(self, name), name)
        for name in ("burn_in", "seed"):
            _integer(getattr(self, name), name, 0)
        for name in ("dt", "duration", "output_interval", "block_multiplier", "min_average_sign"):
            _positive(getattr(self, name), name)
        beads = tuple(_integer(p, "beads") for p in self.beads)
        if not beads or len(set(beads)) != len(beads):
            raise ValueError("beads must contain unique positive integers")
        for value in (self.duration/self.output_interval, self.output_interval/self.dt):
            if not np.isclose(value, round(value), atol=1e-10, rtol=0):
                raise ValueError("dt must divide output interval, which must divide duration")
        if self.min_average_sign > 1:
            raise ValueError("min_average_sign must not exceed one")
        if self.potential_split not in ("lowest_adiabatic", "trace"):
            raise ValueError("potential_split must be lowest_adiabatic or trace")
        object.__setattr__(self, "beads", beads)


def _dynamics(model, beads, dt, split):
    cls = Paper2013Dynamics if split == "lowest_adiabatic" else ThermalNRPMDDynamics
    return cls(model, [1.], 1., beads, dt, 2, native=False, integrator="cayley-bcb")


def propagate_correlations(ensemble, *, dt, duration, output_interval,
                           potential_split="lowest_adiabatic"):
    """Return each original parent's Eq. 12 numerator, before sign weighting."""
    dt, duration, output_interval = (_positive(v, n) for v, n in
        ((dt, "dt"), (duration, "duration"), (output_interval, "output_interval")))
    for value in (duration/output_interval, output_interval/dt):
        if not np.isclose(value, round(value), atol=1e-10, rtol=0):
            raise ValueError("propagation times must share exact observation intervals")
    if potential_split not in ("lowest_adiabatic", "trace"):
        raise ValueError("unknown potential split")
    if (ensemble.beta != 1. or not np.array_equal(ensemble.masses, [1.])
            or not isinstance(ensemble.potential, PaperTwoStateModel)
            or ensemble.sampling_diagnostics.get("static_bias", False)):
        raise ValueError("paper correlations require unbiased unit-mass beta=1 model parents")
    nframes = int(round(duration/output_interval))+1
    steps = int(round(output_interval/dt))
    count = len(ensemble.states)
    dynamics = _dynamics(ensemble.potential, len(ensemble.states[0].q), dt, potential_split)
    numerator = np.empty((count, nframes, 2))
    drift = np.zeros(count)
    final = {name: [] for name in ("q", "p", "mapping_q", "mapping_p")}
    for i, parent in enumerate(ensemble.states):
        state = parent.copy()
        initial_x = float(parent.q[:, 0, 0].mean())
        initial_projector = ensemble.electronic_populations[i, 0]
        energy = dynamics.energy(state)
        for frame in range(nframes):
            if frame:
                for _ in range(steps):
                    dynamics.step(state)
            numerator[i, frame, 0] = initial_x*state.q[:, 0, 0].mean()
            numerator[i, frame, 1] = initial_projector*state.populations()[0]
            drift[i] = max(drift[i], abs(dynamics.energy(state)-energy))
        for name in final:
            final[name].append(getattr(state, name).copy())
    return {"numerators": numerator, "signs": ensemble.signs.copy(),
            "time": np.arange(nframes)*output_interval,
            "maximum_energy_drift_per_parent": drift,
            **{"final_"+name: np.asarray(values) for name, values in final.items()}}


def correlation_bootstrap(numerators, signs, *, replicates=256, seed=0,
                          block_multiplier=2.0, min_average_sign=0.05):
    """Joint ratio bootstrap of independent chains and circular source blocks.

    ``numerators`` has shape [chain, parent, time, observable].  Each chain is
    separately diagnosed before chain resampling.  Sign failures invalidate
    confidence intervals; their draws are never silently discarded to form a
    supposedly valid interval.
    """
    values, signs = np.asarray(numerators, float), np.asarray(signs, float)
    if (values.ndim != 4 or min(values.shape) == 0 or signs.shape != values.shape[:2]
            or not np.isfinite(values).all() or not np.all(np.isin(signs, [-1., 1.]))):
        raise ValueError("invalid chain correlation arrays")
    replicates = _integer(replicates, "replicates")
    block_multiplier = _positive(block_multiplier, "block_multiplier")
    min_average_sign = _positive(min_average_sign, "min_average_sign")
    if min_average_sign > 1:
        raise ValueError("min_average_sign must not exceed one")
    chains, count = signs.shape
    mean_sign = float(signs.mean())
    estimates = (np.mean(values*signs[:, :, None, None], axis=(0, 1))/mean_sign
                 if mean_sign >= min_average_sign else None)
    lengths, inefficiencies = [], []
    for chain in range(chains):
        diagnostics = np.column_stack((signs[chain], (values[chain]*signs[chain, :, None, None]).reshape(count, -1)))
        nonconstant = diagnostics[:, np.var(diagnostics, axis=0) > np.finfo(float).eps]
        g = statistical_inefficiency(nonconstant) if nonconstant.shape[1] else 1.0
        inefficiencies.append(float(g))
        lengths.append(min(count, max(1, int(math.ceil(block_multiplier*g)))))
    rng = np.random.default_rng(seed)
    draws, sign_draws, failures = [], [], []
    for draw in range(replicates):
        chosen_values, chosen_signs = [], []
        for chain in rng.integers(chains, size=chains):
            length = lengths[chain]
            starts = rng.integers(count, size=math.ceil(count/length))
            indices = ((starts[:, None]+np.arange(length)) % count).reshape(-1)[:count]
            chosen_values.append(values[chain, indices])
            chosen_signs.append(signs[chain, indices])
        chosen_values, chosen_signs = np.asarray(chosen_values), np.asarray(chosen_signs)
        normalization = float(chosen_signs.mean())
        sign_draws.append(normalization)
        if normalization < min_average_sign:
            failures.append({"replicate": draw, "average_sign": normalization})
        else:
            draws.append(np.mean(chosen_values*chosen_signs[:, :, None, None], axis=(0, 1))/normalization)
    reasons = []
    if estimates is None:
        reasons.append("source_average_sign_below_threshold")
    if chains < 2:
        reasons.append("fewer_than_two_independent_chains")
    if replicates < 32:
        reasons.append("fewer_than_32_bootstrap_replicates")
    if min(count//length for length in lengths) < 8:
        reasons.append("fewer_than_eight_complete_blocks_in_a_source_chain")
    if failures:
        reasons.append("bootstrap_contains_unresolved_signed_normalization")
    sign_error = float(np.std(sign_draws, ddof=1)) if replicates > 1 else None
    if sign_error is not None and mean_sign <= 2*sign_error:
        reasons.append("source_average_sign_not_resolved_against_bootstrap_error")
    available = not reasons
    draws = np.asarray(draws)
    return {"method": "hierarchical_source_chain_circular_block_joint_ratio_bootstrap",
            "scope": "conditional_on_source_mixing_and_selected_block_lengths",
            "estimate": None if estimates is None else estimates.tolist(),
            "average_sign": mean_sign, "average_sign_standard_error": sign_error,
            "independent_chain_average_signs": signs.mean(axis=1).tolist(),
            "independent_chain_estimates": [((values[c]*signs[c, :, None, None]).mean(axis=0)/signs[c].mean()).tolist()
                if signs[c].mean() >= min_average_sign else None for c in range(chains)],
            "statistical_inefficiency_diagnostics": inefficiencies, "block_lengths": lengths,
            "minimum_complete_blocks_per_chain": min(count//length for length in lengths),
            "requested_replicates": replicates, "valid_replicates": len(draws),
            "failure_records": failures, "confidence_intervals_available": available,
            "confidence_level": 0.95, "unavailable_reasons": reasons,
            "standard_error": np.std(draws, axis=0, ddof=1).tolist() if available else None,
            "percentile_interval": np.quantile(draws, [0.025, 0.975], axis=0).tolist() if available else None,
            "physical_converged": False}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _canonical(value):
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def _identity(config):
    root = Path(thermal_module.__file__).resolve().parents[2]
    paths = [Path(__file__).resolve()]+[root/"src/pyrpmd"/name for name in
             ("thermal_nrpmd.py", "nonadiabatic.py", "dynamics.py", "analysis/fes.py")]
    return {"config": asdict(config), "source_sha256":
            {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}


def _run_chain(config_dict, beads, chain, output_directory, resume=False):
    config = BenchmarkConfig(**config_dict)
    output = Path(output_directory)
    name = f"P{beads}_chain{chain}"
    report_path, array_path = output/(name+".json"), output/(name+".npz")
    identity = {**_identity(config), "beads": beads, "chain": chain}
    fingerprint = hashlib.sha256(_canonical(identity).encode()).hexdigest()
    if report_path.exists():
        saved = json.loads(report_path.read_text())
        if not resume or saved.get("fingerprint") != fingerprint:
            raise ValueError(f"existing chain requires --resume and identical config/sources: {name}")
        if not array_path.is_file() or hashlib.sha256(array_path.read_bytes()).hexdigest() != saved["array_sha256"]:
            raise ValueError(f"missing or modified chain arrays: {name}")
        return saved
    with threadpool_limits(limits=1):
        start = time.perf_counter()
        model = PaperTwoStateModel()
        sampler = ThermalNRPMDSampler(
            model, [1.], 1., np.zeros((beads, 1, 3)), 2,
            seed=np.random.SeedSequence([config.seed, beads, chain]),
            nuclear_step=0.5/math.sqrt(beads), centroid_step=0.65,
        )
        ensemble = sampler.sample(config.samples, burn_in=config.burn_in, stride=config.stride)
        sampled_seconds = time.perf_counter()-start
        data = propagate_correlations(ensemble, dt=config.dt, duration=config.duration,
            output_interval=config.output_interval, potential_split=config.potential_split)
        for key in ("q", "p", "mapping_q", "mapping_p"):
            data["initial_"+key] = np.array([getattr(state, key) for state in ensemble.states])
        data["electronic_populations"] = ensemble.electronic_populations
        data["sweeps"] = ensemble.sweeps
        # Numerical refinement is paired on the first fixed subset, never on
        # a newly sampled or favorable subset.  It is not a sampling error bar.
        from pyrpmd.thermal_nrpmd import ThermalEnsemble
        keep = min(config.refinement_parents, config.samples)
        subset = ThermalEnsemble(ensemble.states[:keep], ensemble.signs[:keep],
            ensemble.electronic_populations[:keep], ensemble.masses, ensemble.beta,
            ensemble.potential, ensemble.sweeps[:keep], ensemble.sampling_diagnostics)
        refined = propagate_correlations(subset, dt=config.dt/2, duration=config.duration,
            output_interval=config.output_interval, potential_split=config.potential_split)
        data["refined_numerators"] = refined["numerators"]
        data["refinement_source_indices"] = np.arange(keep)
        refinement = {"parents": keep, "same_complete_initial_states": True,
            "coarse_dt": config.dt, "reference_dt": config.dt/2,
            "same_physical_duration": config.duration,
            "maximum_absolute_per_parent_correlation_difference":
                np.max(abs(data["numerators"][:keep]-refined["numerators"]), axis=(0, 1)).tolist(),
            "maximum_absolute_energy_drift_coarse": float(data["maximum_energy_drift_per_parent"][:keep].max()),
            "maximum_absolute_energy_drift_fine": float(refined["maximum_energy_drift_per_parent"].max()),
            "uncertainty_scope": "paired_numerical_difference_not_sampling_uncertainty"}
        temporary = array_path.with_name(array_path.name+".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **data)
        temporary.replace(array_path)
        report = {**identity, "fingerprint": fingerprint, "arrays": array_path.name,
            "array_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
            "sampling": ensemble.sampling_diagnostics,
            "average_sign": float(ensemble.signs.mean()),
            "timestep_refinement": refinement,
            "maximum_absolute_energy_drift": float(data["maximum_energy_drift_per_parent"].max()),
            "timing_seconds": {"sampling": sampled_seconds, "total": time.perf_counter()-start}}
        _write_json(report_path, report)
        return report


def compare_published_data(path, reference, results):
    """Compare only explicitly digitized rows; no fabricated paper numbers.

    CSV columns: observable (Cxx/C11), series (dvr/P1/P4/P8), time,
    value, absolute_uncertainty.  Digitization uncertainty is not added to
    Monte Carlo uncertainty or treated as an original paper confidence band.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    dvr_times = sorted({float(row["time"]) for row in rows if row["series"] == "dvr"})
    # Evaluating the spectral sum at the digitized abscissae avoids adding a
    # time-interpolation error to the independent exact-reference comparison.
    dvr_at_published_times = None
    if dvr_times:
        with threadpool_limits(limits=1):
            dvr_at_published_times = exact_kubo(dvr_times, **{
                name: reference[name] for name in ("alpha", "kappa", "delta", "beta", "points", "extent")})
    comparisons = []
    for row in rows:
        name, series = row["observable"], row["series"]
        if name not in ("Cxx", "C11"):
            raise ValueError("unknown published observable")
        t, published = float(row["time"]), float(row["value"])
        uncertainty = _positive(float(row["absolute_uncertainty"]), "absolute_uncertainty", zero=True)
        if not np.isfinite([t, published]).all():
            raise ValueError("published points must be finite")
        if series == "dvr":
            timeline, curve, se = reference["time"], reference[name], None
        else:
            if series not in ("P1", "P4", "P8"):
                raise ValueError("unknown published series")
            available = {f"P{r['beads']}": r for r in results}
            if series not in available:
                continue
            result = available[series]
            bootstrap = result["statistics"]
            if bootstrap["estimate"] is None:
                continue
            column = ("Cxx", "C11").index(name)
            timeline = result["time"]
            curve = np.asarray(bootstrap["estimate"])[:, column]
            se = np.asarray(bootstrap["standard_error"])[:, column] if bootstrap["confidence_intervals_available"] else None
        if t < timeline[0]-1e-10 or t > timeline[-1]+1e-10:
            continue
        computed = (float(dvr_at_published_times[name][dvr_times.index(t)]) if series == "dvr"
                    else float(np.interp(t, timeline, curve)))
        comparisons.append({"observable": name, "series": series, "time": t,
            "published_digitized_value": published, "digitization_absolute_uncertainty": uncertainty,
            "computed_value": computed, "difference": computed-published,
            "computed_bootstrap_standard_error": None if se is None else float(np.interp(t, timeline, se)),
            "computed_curve_interpolation": ("none_spectral_sum_at_published_time" if series == "dvr"
                                               else "linear_between_saved_times")})
    summaries = []
    for series in sorted({row["series"] for row in comparisons}):
        for name in ("Cxx", "C11"):
            selected = [row for row in comparisons if row["series"] == series and row["observable"] == name]
            if selected:
                difference = np.array([row["difference"] for row in selected])
                summaries.append({"series": series, "observable": name, "points": len(selected),
                    "rmse": float(np.sqrt(np.mean(difference**2))),
                    "maximum_absolute_difference": float(np.max(abs(difference)))})
    return {"available": bool(comparisons), "source_csv": str(path), "source_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "points": comparisons, "summary": summaries,
            "scope": "direct_comparison_to_digitized_published_figure_with_separate_digitization_error"}


def run_benchmark(config, output, *, workers=1, resume=False, published_data=None):
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be BenchmarkConfig")
    workers = _integer(workers, "workers")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = _identity(config)
    manifest = output/"plan.json"
    if manifest.exists():
        if not resume or _canonical(json.loads(manifest.read_text())) != _canonical(identity):
            raise ValueError("existing benchmark requires --resume with identical configuration and sources")
    else:
        _write_json(manifest, identity)
    tasks = [(p, c) for p in config.beads for c in range(config.chains)]
    reports = []
    if workers == 1:
        for beads, chain in tasks:
            reports.append(_run_chain(asdict(config), beads, chain, str(output), resume))
            print(f"completed P{beads} chain {chain}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_chain, asdict(config), p, c, str(output), resume): (p, c) for p, c in tasks}
            for future in as_completed(futures):
                reports.append(future.result())
                p, c = futures[future]
                print(f"completed P{p} chain {c}", flush=True)
    reports.sort(key=lambda r: (config.beads.index(r["beads"]), r["chain"]))
    times = np.arange(int(round(config.duration/config.output_interval))+1)*config.output_interval
    with threadpool_limits(limits=1):
        reference = exact_kubo(times, points=161, extent=9.)
        checks = [exact_kubo(times, points=129, extent=9.), exact_kubo(times, points=181, extent=10.)]
    reference["convergence_checks"] = [{"points": check["points"], "extent": check["extent"],
        "maximum_absolute_curve_difference": {name: float(np.max(abs(np.asarray(reference[name])-check[name])))
                                              for name in ("Cxx", "C11")}} for check in checks]
    results = []
    for beads in config.beads:
        selected = [report for report in reports if report["beads"] == beads]
        values, signs = [], []
        for report in selected:
            with np.load(output/report["arrays"], allow_pickle=False) as arrays:
                values.append(arrays["numerators"])
                signs.append(arrays["signs"])
        bootstrap = correlation_bootstrap(values, signs, replicates=config.bootstrap_replicates,
            seed=np.random.SeedSequence([config.seed, beads, 999]),
            block_multiplier=config.block_multiplier, min_average_sign=config.min_average_sign)
        metrics = None
        if bootstrap["estimate"] is not None:
            estimate = np.asarray(bootstrap["estimate"])
            metrics = {name: {"rmse": float(np.sqrt(np.mean((estimate[:, i]-reference[name])**2))),
                               "maximum_absolute_difference": float(np.max(abs(estimate[:, i]-reference[name]))),
                               "zero_time_difference": float(estimate[0, i]-reference[name][0])}
                       for i, name in enumerate(("Cxx", "C11"))}
        results.append({"beads": beads, "time": times.tolist(), "statistics": bootstrap,
                        "comparison_to_recomputed_exact_quantum": metrics, "chains": selected})
    report = {"schema_version": 1, "paper": PAPER, **identity,
        "model": {"alpha": 0., "kappa": 1., "delta": 1., "beta": 1., "mass": 1., "hbar": 1., "omega": 1.},
        "observable_order": ["Cxx", "C11"], "time_units": "inverse_omega",
        "potential_split": config.potential_split,
        "paper_potential_split": "lowest_adiabatic",
        "paper_algorithm_partition_matched": config.potential_split == "lowest_adiabatic",
        "integrator": "existing_MInt_plus_Cayley_free_ring_symmetric_splitting",
        "spectator_coordinates": "separable_harmonic_y_and_z_cancel_from_normalized_x_and_electronic_quantities",
        "exact_quantum_reference": reference, "results": results,
        "comparison_scope": "2013_paper_nonadiabatic_model_not_validation_of_electrochemical_reservoir_or_adaptive_sampling",
        "physical_converged": False, "rate_constant": None}
    report["published_figure_comparison"] = (compare_published_data(published_data, reference, results)
        if published_data is not None else {"available": False, "reason": "no_digitized_published_figure_supplied"})
    _write_json(output/"summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/published-nrpmd-benchmark"))
    parser.add_argument("--beads", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--burn-in", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--duration", type=float, default=5.)
    parser.add_argument("--output-interval", type=float, default=0.1)
    parser.add_argument("--bootstrap-replicates", type=int, default=256)
    parser.add_argument("--refinement-parents", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--potential-split", choices=["lowest_adiabatic", "trace"], default="lowest_adiabatic")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--published-data", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    config = BenchmarkConfig(**{name: getattr(args, name) for name in (
        "beads", "chains", "samples", "burn_in", "stride", "dt", "duration", "output_interval",
        "bootstrap_replicates", "refinement_parents", "seed", "potential_split")})
    if args.plan:
        print(json.dumps(_jsonable({"paper": PAPER, **_identity(config),
            "total_parents": len(config.beads)*config.chains*config.samples,
            "production_trajectory_steps": int(round(config.duration/config.dt))*len(config.beads)*config.chains*config.samples}), indent=2))
        return
    run_benchmark(config, args.output, workers=args.workers, resume=args.resume, published_data=args.published_data)


if __name__ == "__main__":
    main()
