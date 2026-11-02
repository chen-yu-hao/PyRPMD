"""Reproducible condensed-phase SC/PIGLET/RPC validation.

Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:
python benchmarks/condensed_phase/validate.py --case harmonic --output <directory>
python benchmarks/condensed_phase/validate.py --case lj --output <directory>

Independent seeds are the units for reported MD confidence intervals. A short
run is a validation pilot, not a general assertion of thermodynamic convergence.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import platform
import inspect
import sys
import time

import numpy as np
from scipy.integrate import simpson
from scipy.stats import t as student_t

from models import HarmonicCrystal, LennardJonesSolid, build_lj, fcc, piglet_parameters
from pyrpmd.barostat import BZPBarostat
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.pimd_methods import SuzukiChinPotential
from pyrpmd.potentials.contraction import SplitPotential
from pyrpmd.suzuki_chin import suzuki_chin_thermodynamics


def finite_bead_crystal(model, beta, beads, *, sc=True, volume=None):
    """Independent Gaussian determinant derivatives; no production estimators."""
    p = beads
    volume = model.volume if volume is None else volume
    stiffness = model.eigenvalues*(volume/model.reference_volume)**(-2*model.gamma)
    ring = 2*np.eye(p)-np.roll(np.eye(p), 1, axis=0)-np.roll(np.eye(p), -1, axis=0)
    weights = np.where(np.arange(p)%2 == 0, 2/3, 4/3) if sc else np.ones(p)
    wd = np.where(np.arange(p)%2 == 0, 0., 1/9) if sc else np.zeros(p)
    precision, covariance = [], []
    energy, cv, logz = model.elastic(volume), 0., -beta*model.elastic(volume)
    for k in stiffness:
        a = model.mass*p*ring
        b = np.diag(weights*k/p)
        c = np.diag(2*wd*k*k/(model.mass*p**3))
        matrix = a/beta + b*beta + c*beta**3
        inverse = np.linalg.inv(matrix)
        first = -a/beta**2+b+3*c*beta**2
        second = 2*a/beta**3+6*c*beta
        energy += 1.5*(p/beta+np.trace(inverse@first))
        cv += 1.5*(p-beta**2*np.trace(inverse@second-inverse@first@inverse@first))
        logz += 1.5*(p*np.log(model.mass*p/beta)-np.linalg.slogdet(matrix)[1])
        precision.append(matrix)
        covariance.append(inverse)
    pressure = model.gamma*(energy-model.elastic(volume))/volume-model.bulk*(volume-model.reference_volume)/model.reference_volume
    return dict(energy=float(energy), heat_capacity_kB=float(cv), pressure=float(pressure),
                log_partition=float(logz), covariance=np.array(covariance))


def npt_crystal_reference(model, beta, pressure, beads=None):
    """One-dimensional volume quadrature of independent Gaussian partition Z."""
    volumes = np.linspace(.55, 1.6, 1001)*model.reference_volume
    refs = [(model.quantum_reference(beta, v) if beads is None else
             finite_bead_crystal(model, beta, beads, volume=v)) for v in volumes]
    logweight = np.array([r["log_partition"] for r in refs])-beta*pressure*volumes
    weights = np.exp(logweight-logweight.max())
    norm = simpson(weights, x=volumes)
    mean = lambda x: float(simpson(weights*np.asarray(x), x=volumes)/norm)
    enthalpy = np.array([r["energy"] for r in refs])+pressure*volumes
    mean_h = mean(enthalpy)
    return {"volume": mean(volumes), "volume_second_moment": mean(volumes**2),
            "energy": mean([r["energy"] for r in refs]),
            "pressure": mean([r["pressure"] for r in refs]),
            "enthalpy": mean_h,
            "heat_capacity_Cp_kB": mean([r["heat_capacity_kB"] for r in refs])+beta**2*mean((enthalpy-mean_h)**2),
            "relative_endpoint_weight": float(max(weights[0], weights[-1])/weights.max())}


def gaussian_estimator_moments(model, beta, beads):
    """Exactly integrate production quadratic estimators over the crystal.

    Recover each polynomial from basis probes, then use Gaussian Wick moments.
    This independently tests the estimator implementation without Monte Carlo
    noise. The partition-function reference uses matrix derivatives instead.
    Degenerate spatial modes share one probe; there are four distinct bands.
    """
    action = SuzukiChinPotential(model, model.masses, beta, beads)
    reference = finite_bead_crystal(model, beta, beads)
    indices = list(zip(*np.triu_indices(beads, 1)))
    vectors = np.concatenate([np.zeros((1, beads)), np.eye(beads),
                              np.array([np.eye(beads)[i]+np.eye(beads)[j] for i,j in indices])])
    expectation = {"internal_energy_hartree": 1.5*model.natoms/beta+model.elastic(),
                   "scaled_action_second_beta_derivative": 0.,
                   "pressure_au": (model.natoms/beta-model.volume*model.bulk*(model.volume-model.reference_volume)/model.reference_volume)/model.volume}
    variance = 0.
    values, counts = np.unique(np.round(model.eigenvalues, 12), return_counts=True)
    for eigenvalue, count in zip(values, counts):
        index = int(np.argmin(abs(model.eigenvalues-eigenvalue)))
        displacement = np.zeros((len(vectors), beads, model.natoms, 3))
        displacement[..., 0] = vectors[:, :, None]*model.eigenvectors[:, index]
        observed = action.estimators_batch(displacement+model.centers, volumes=model.volume)
        covariance = reference["covariance"][index]
        for key in expectation:
            series = observed[key]
            diagonal = series[1:1+beads]-series[0]
            polynomial = np.diag(diagonal)
            for value, (i,j) in zip(series[1+beads:], indices):
                polynomial[i,j] = polynomial[j,i] = .5*(value-series[0]-diagonal[i]-diagonal[j])
            expectation[key] += 3*count*float(np.trace(polynomial@covariance))
            if key == "internal_energy_hartree":
                variance += 6*count*float(np.trace(polynomial@covariance@polynomial@covariance))
    return {"energy": expectation["internal_energy_hartree"],
            "heat_capacity_kB": 1.5*model.natoms+beta**2*(variance-expectation["scaled_action_second_beta_derivative"]),
            "pressure": expectation["pressure_au"],
            "method": "quadratic production-estimator basis probes + exact Gaussian Wick moments"}


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
    radius = None if se is None else float(student_t.ppf(.975, len(values)-1)*se)
    return {"mean": mean, "standard_error": se,
            "confidence_interval_95": None if radius is None else [mean-radius, mean+radius],
            "independent_replicas": len(values)}


def primitive_observables(provider, q, beta, volume):
    physical = provider.evaluate(q)
    offsets = q-q.mean(axis=0, keepdims=True)
    correction = float(np.sum(offsets*physical.forces)/len(q))
    kinetic = 1.5*q.shape[1]/beta-.5*correction
    trace = float(np.trace(physical.virial, axis1=1, axis2=2).mean())
    return dict(internal_energy_hartree=float(physical.energy.mean()+kinetic),
                kinetic_energy_hartree=kinetic,
                potential_energy_hartree=float(physical.energy.mean()),
                pressure_au=(3*q.shape[1]/beta+trace-correction)/(3*volume))


def save_csv(path, rows):
    keys = list(rows[0])
    data = np.array([[row[k] for k in keys] for row in rows])
    np.savetxt(path, data, delimiter=",", header=",".join(keys), comments="", fmt="%.17g")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_source_identity():
    """Stable identities of numerical definitions, independent of report edits."""
    root = Path(__file__).resolve().parents[2]
    files = ["benchmarks/condensed_phase/models.py", "benchmarks/condensed_phase/lj.cpp",
             "benchmarks/condensed_phase/piglet_4_750K.json",
             *["src/pyrpmd/"+name for name in ("dynamics.py", "engine.py", "barostat.py",
               "native_ensemble.py", "pimd_methods.py", "suzuki_chin.py", "constants.py",
               "potentials/contraction.py")]]
    hashes = {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
    for function in (run_trajectory, primitive_observables):
        tree = ast.dump(ast.parse(inspect.getsource(function)), include_attributes=False)
        hashes["validate.py::"+function.__name__+"::AST"] = hashlib.sha256(tree.encode()).hexdigest()
    import importlib.util
    spec = importlib.util.find_spec("pyrpmd._native")
    if spec is not None:
        hashes["pyrpmd._native compiled binary"] = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
    return hashes


def trajectory_identity(args, method, beads, dt, pressure, replica):
    return {"schema":"condensed-trajectory-identity-v1", "model":"smooth LJ FCC32",
            "method":method, "beads":beads, "dt":dt, "external_pressure":pressure,
            "seed":82719+replica,"burn":args.burn,"steps":args.steps,"stride":args.stride,
            "beta":5.,"mass":100.,"density":1.,"switch_start":2.,"cutoff":2.5,
            "tau":5.,"contracted_beads":({"rpc":4,"rpc8":8}.get(method)),
            "numerical_source_sha256":numerical_source_identity()}


def save_provenance(path, identity, *, recording="captured_before_execution"):
    metadata={"identity":identity,"recording":recording,
              "raw_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".provenance.json").write_text(json.dumps(metadata,indent=2)+"\n")


def load_csv(path, *, burn, steps, stride, identity):
    provenance = path.with_suffix(".provenance.json")
    if not provenance.exists():
        raise ValueError(f"cannot resume trajectory without protocol/source provenance: {path}")
    metadata = json.loads(provenance.read_text())
    if metadata["identity"] != identity or metadata["raw_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError(f"saved trajectory source/protocol/content identity differs: {path}")
    values = np.genfromtxt(path, delimiter=",", names=True)
    values = np.atleast_1d(values)
    expected = np.arange(burn+stride, burn+steps+1, stride)
    if not np.array_equal(values["step"], expected):
        raise ValueError(f"saved trajectory clock/count disagree with requested protocol: {path}")
    return [{key:float(row[key]) for key in values.dtype.names} for row in values]


def run_trajectory(provider, geometry, cell, masses, beta, beads, dt, seed,
                   burn, steps, stride, method, *, pressure=None, tau=3.):
    action = SuzukiChinPotential(provider, masses, beta, beads) if method == "sc" else provider
    thermostat = {"mode": "piglet", "gle": piglet_parameters(beta)} if method == "piglet" else "pile-l"
    system = {"cell": cell, "pbc": [True]*3, "masses": masses}
    piston = None if pressure is None else BZPBarostat(action, system, beta, beads, pressure, tau)
    dynamics = Dynamics(action, masses, beta, beads, dt, seed=seed, native=True,
                        integrator="normal-mode-bab", loop_backend="python",
                        thermostat=thermostat, barostat=piston, cell=cell, pbc=[True]*3)
    state = State(np.repeat(geometry[None], beads, axis=0), dynamics.sample_momenta())
    records = []
    for step in range(burn+steps):
        dynamics.step(state)
        if step >= burn and (step-burn+1)%stride == 0:
            current_cell = cell if piston is None else piston.cell
            volume = float(np.linalg.det(current_cell))
            obs = (action.estimators(state.q, volume=volume) if method == "sc" else
                   primitive_observables(provider, state.q, beta, volume))
            # Translational drift is removed only in this structural diagnostic,
            # never in the dynamics or pressure/energy normalization.
            displacement = state.q.mean(axis=0) - geometry@np.linalg.solve(cell, current_cell)
            displacement -= displacement.mean(axis=0)
            lindemann = float(np.sqrt(np.mean(np.sum(displacement**2, axis=1))))
            records.append(dict(step=float(step+1), volume=volume,
                                centroid_rms_displacement=lindemann, **obs))
    return records


def harmonic(args):
    model = HarmonicCrystal(bulk=.2)
    beta = 8.
    exact = model.quantum_reference(beta)
    result = {"model": "27-atom pinned coupled harmonic crystal, periodic nearest-neighbour graph",
              "parameters": dict(beta=beta, atoms=model.natoms, mass=model.mass,
                                 gamma=model.gamma, volume=model.volume, bulk=model.bulk),
              "quantum_reference": exact, "gaussian_estimator_cases": [], "npt": []}
    for beads in (4, 8, 16, 32):
        reference = finite_bead_crystal(model, beta, beads)
        action = SuzukiChinPotential(model, model.masses, beta, beads)
        rng = np.random.default_rng(7139+beads)
        roots = np.linalg.cholesky(reference.pop("covariance"))
        columns = []
        for start in range(0, args.samples, 256):
            size = min(256, args.samples-start)
            z = rng.normal(size=(size, model.natoms, beads, 3))
            modes = np.einsum("nij,snja->snia", roots, z)
            paths = np.einsum("in,snpa->spia", model.eigenvectors, modes)+model.centers
            columns.append(action.estimators_batch(paths, volumes=model.volume))
        series = {key: np.concatenate([block[key] for block in columns]) for key in columns[0]}
        report = suzuki_chin_thermodynamics(series, beta, block_size=256)
        case = {"beads": beads, "samples": args.samples, "seed": 7139+beads,
                "finite_bead_reference": reference, "production_estimators": report,
                "deterministic_production_moments": gaussian_estimator_moments(model, beta, beads)}
        result["gaussian_estimator_cases"].append(case)
        print(json.dumps({"event":"harmonic_gaussian", "beads":beads,
                          "energy":report["internal_energy_hartree"], "energy_ref":reference["energy"],
                          "cv":report["heat_capacity_over_kB"], "cv_ref":reference["heat_capacity_kB"]}), flush=True)
    external = exact["pressure"]
    for beads, dt in ((8, .04), (8, .02)):
        target = npt_crystal_reference(model, beta, external, beads)
        cases = []
        for replica in range(args.replicas):
            model.set_cell(model.reference_cell)
            rows = run_trajectory(model, model.centers.copy(), model.cell.copy(), model.masses,
                                  beta, beads, dt, 18017+replica, args.burn, args.steps,
                                  args.stride, "sc", pressure=external, tau=2.)
            path = args.output/f"crystal_sc_npt_p{beads}_dt{dt}_seed{18017+replica}.csv"
            digest = save_csv(path, rows)
            means = {k: float(np.mean([r[k] for r in rows])) for k in
                     ("volume", "pressure_au", "internal_energy_hartree")}
            cases.append({"seed":18017+replica, "means":means, "raw":path.name, "sha256":digest})
            print(json.dumps({"event":"harmonic_npt", "beads":beads, "dt":dt, "replica":replica, **means}), flush=True)
        result["npt"].append({"beads":beads, "dt":dt, "reference":target,
                              "replicas":cases, "statistics":{k:interval([c["means"][k] for c in cases])
                              for k in cases[0]["means"]}})
    return result


def lj(args):
    library = build_lj(args.output/"build")
    geometry, cell = fcc(cells=2, density=1.0)
    beta, mass = 5., 100.
    masses = np.full(len(geometry), mass)
    result = {"model":"32-atom FCC Lennard-Jones solid, all periodic images within a smooth 2.5 sigma cutoff",
              "units":"epsilon=sigma=hbar=1; mass=100, quantum parameter=0.1",
              "parameters":dict(beta=beta, atoms=len(geometry), mass=mass, initial_density=1.,
                                 cutoff=2.5, switch_start=2.0), "cases":[]}
    # Full primitive rings provide a separate discretization reference; SC
    # bead and timestep refinements test convergence without fitting to data.
    protocols = [("primitive", 32, .04, None), ("primitive", 64, .04, None),
                 ("sc", 4, .04, None), ("sc", 8, .04, None), ("sc", 8, .02, None),
                 ("sc", 8, .04, 1.), ("sc", 8, .02, 1.), ("piglet", 4, .04, 1.),
                 ("primitive", 32, .04, 1.), ("rpc", 16, .04, 1.),
                 ("sc", 16, .04, None), ("sc", 32, .04, None),
                 ("sc", 16, .04, 1.), ("primitive", 16, .04, 1.),
                 ("rpc8", 16, .04, 1.)]
    if args.quick:
        protocols = [("sc", 4, .04, None), ("sc", 8, .04, 1.), ("piglet", 4, .04, 1.)]
    if args.protocol_index is not None:
        protocols = [protocols[args.protocol_index]]
    elif args.jobs > 1:
        # Protocols have distinct filenames. Each child owns complete replicas
        # of one protocol and its own partial summary; the parent joins them.
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(lj_worker, vars(args), index):index for index in range(len(protocols))}
            completed = {}
            for future in concurrent.futures.as_completed(futures):
                completed[futures[future]] = future.result()
        result["cases"] = [case for index in sorted(completed) for case in completed[index]["cases"]]
        return result
    for method, beads, dt, pressure in protocols:
        cases = []
        for replica in range(args.replicas):
            physical = LennardJonesSolid(library, len(geometry), cell.copy())
            provider = physical
            if method in ("rpc", "rpc8"):
                fast = LennardJonesSolid(library, len(geometry), cell.copy(), component=1)
                slow = LennardJonesSolid(library, len(geometry), cell.copy(), component=2)
                provider = SplitPotential(fast, slow, beads, 4 if method == "rpc" else 8)
            ensemble = "nvt" if pressure is None else "npt"
            path = args.output/f"lj_{method}_{ensemble}_p{beads}_dt{dt}_seed{82719+replica}.csv"
            identity = trajectory_identity(args, method, beads, dt, pressure, replica)
            if args.resume and path.exists():
                rows = load_csv(path, burn=args.burn, steps=args.steps, stride=args.stride, identity=identity)
            else:
                rows = run_trajectory(provider, geometry, cell, masses, beta, beads, dt,
                                      82719+replica, args.burn, args.steps, args.stride, method,
                                      pressure=pressure, tau=5.)
                if identity["numerical_source_sha256"] != numerical_source_identity():
                    raise RuntimeError("numerical source changed during trajectory; do not adopt mixed-version evidence")
            digest = save_csv(path, rows)
            if not path.with_suffix(".provenance.json").exists():
                save_provenance(path, identity)
            keys = ("volume", "pressure_au", "internal_energy_hartree", "kinetic_energy_hartree",
                    "potential_energy_hartree", "centroid_rms_displacement")
            means = {k:float(np.mean([r[k] for r in rows])) for k in keys}
            row = {"seed":82719+replica,"means":means,"raw":path.name,"sha256":digest}
            if method == "sc":
                row["thermodynamics"] = suzuki_chin_thermodynamics(rows, beta,
                    volumes=None if pressure is None else [r["volume"] for r in rows],
                    pressure_au=pressure, block_size=max(1,len(rows)//20))
            cases.append(row)
            print(json.dumps({"event":"lj", "method":method,"beads":beads,"dt":dt,
                              "ensemble":ensemble,"replica":replica, **means}),flush=True)
        result["cases"].append(dict(method=method,beads=beads,dt=dt,external_pressure=pressure,
                                    replicas=cases,statistics={k:interval([c["means"][k] for c in cases]) for k in keys}))
        # Preserve completed protocols if a later case is interrupted.
        write_report(args, result)
    return result


def lj_worker(options, index):
    args = argparse.Namespace(**options)
    args.protocol_index, args.jobs = index, 1
    return lj(args)


def write_report(args, result):
    data={"schema":"PyRPMD.condensed-phase-validation.v1","case":args.case,
          "run_options":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
          "python":platform.python_version(),"numpy":np.__version__,
          "convergence_claim":"finite-size model validation; no claim for water or a real material",
          "results":result}
    args.output.mkdir(parents=True,exist_ok=True)
    suffix = "" if args.protocol_index is None else f"_case_{args.protocol_index}"
    (args.output/(args.case+suffix+"_summary.json")).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=("harmonic","lj"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=8192)
    parser.add_argument("--replicas",type=int,default=4)
    parser.add_argument("--burn",type=int,default=5000)
    parser.add_argument("--steps",type=int,default=20000)
    parser.add_argument("--stride",type=int,default=20)
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--jobs",type=int,default=1,help="independent LJ protocols to run concurrently")
    parser.add_argument("--protocol-index",type=int,default=None,help=argparse.SUPPRESS)
    parser.add_argument("--resume",action="store_true",help="reuse complete CSVs with matching saved step clocks")
    args=parser.parse_args()
    if min(args.samples,args.replicas,args.burn,args.steps,args.stride)<1 or args.steps<args.stride:
        parser.error("counts must be positive and steps >= stride")
    args.output.mkdir(parents=True,exist_ok=True)
    result=(harmonic if args.case=="harmonic" else lj)(args)
    write_report(args,result)


if __name__ == "__main__":
    main()
