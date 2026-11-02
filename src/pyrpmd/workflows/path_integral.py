"""Restartable SC, exchange and PIGS sampling through Dynamics."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from ..checkpoint import atomic_savez, load_checkpoint
from ..dynamics import Dynamics, State
from ..pimd_methods import (SuzukiChinPotential, BosonicExchangePotential,
    GaussianTrialWavefunction, PIGSPotential)
from ..potentials import make_potential
from ..potentials.base import validate_result
from ..propagation import loop_options
from ..units import parse_quantity
from ..validation import checked_integer, finite_real
from .fes import _fingerprint
from .materials import resolve_materials


KIND = "PyRPMD.path-integral.v1"
EXCHANGE_METHODS = ("bosonic", "fermionic")
EXCHANGE_COLUMNS = {"sign", "potential_energy_numerator_hartree",
    "kinetic_energy_numerator_hartree", "internal_energy_numerator_hartree"}


def _exchange_statistics_options(table):
    """Validate reporting controls without inserting defaults into run identity."""
    block = table.get("block_size")
    if "block_size" in table:
        block = checked_integer(block, "path_integral.block_size")
    minimum = finite_real(table.get("minimum_average_sign", .05),
                          "path_integral.minimum_average_sign", positive=True)
    if minimum > 1:
        raise ValueError("path_integral.minimum_average_sign must be at most 1")
    sigma = finite_real(table.get("sign_sigma", 3.), "path_integral.sign_sigma", positive=True)
    return dict(block_size=block, minimum_average_sign=minimum, sign_sigma=sigma)


def _exchange_history(value, count, method, pbc):
    """Validate stored signed numerators; no division by instantaneous sign."""
    required = EXCHANGE_COLUMNS | ({"virial_internal_energy_numerator_hartree"} if not any(pbc) else set())
    allowed = required | ({"pressure_numerator_au"} if all(pbc) else set())
    if (not isinstance(value, list) or len(value) != count
            or any(not isinstance(row, dict) or not required <= set(row)
                   or set(row) - allowed for row in value)
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x)
                   for row in value for x in row.values())
            or any(set(row) != set(value[0]) for row in value)
            or any(abs(row["sign"]) > 1 + 64 * np.finfo(float).eps
                   or (method == "bosonic" and row["sign"] != 1.)
                   or not np.isclose(row["internal_energy_numerator_hartree"],
                                     row["potential_energy_numerator_hartree"]
                                     + row["kinetic_energy_numerator_hartree"], rtol=1e-12, atol=1e-14)
                   for row in value)):
        raise ValueError("invalid exchange estimator checkpoint history")
    return value


def resolve_path_integral(config, system):
    table = config.get("path_integral", {})
    if not isinstance(table, dict):
        raise ValueError("path_integral must be a table")
    method = table.get("method", "suzuki_chin")
    fields = {
        "suzuki_chin": {"method", "alpha", "fd_step", "beta_fd_step"},
        "bosonic": {"method", "groups", "periodic_mode", "block_size"},
        "fermionic": {"method", "groups", "periodic_mode", "block_size",
                      "minimum_average_sign", "sign_sigma"},
        "pigs": {"method", "projection_time", "trial"},
    }
    if not isinstance(method, str) or method not in fields or set(table) - fields[method]:
        raise ValueError("Unknown path_integral method or fields for that method")
    if config.get("task") != "pimd":
        raise ValueError("path_integral sampling actions require task='pimd'")
    unsupported = {"active_learning", "electronic_friction", "nonadiabatic", "plumed"}.intersection(config)
    if unsupported:
        raise ValueError("path_integral does not support sections: " + ", ".join(sorted(unsupported)))
    if system.fixed_atoms or system.bond_constraints:
        raise ValueError("SC/exchange/PIGS workflow requires an unconstrained all-mobile system")
    if method == "pigs" and (any(system.pbc) or config.get("barostat")):
        raise ValueError("PIGS workflow requires a nonperiodic fixed-cell system")
    if method in EXCHANGE_METHODS:
        _exchange_statistics_options(table)
        if "barostat" in config:
            raise ValueError("exchange + BZP NPT is unsupported: centroid-only cell dilation does not "
                "preserve the all-bead minimum-image measure; use fixed-cell exchange sampling")
        if any(system.pbc) and loop_options(config)["integrator"] != "velocity-verlet":
            raise ValueError("periodic exchange requires dynamics.integrator='velocity-verlet' "
                "and sampling.thermostat='andersen'; raw normal-mode springs are not lattice invariant")
    if method == "suzuki_chin" and any(system.pbc) and not all(system.pbc):
        raise ValueError("periodic SC requires a fully periodic 3D cell")
    if config.get("sampling", {}).get("engine", "legacy") != "legacy":
        raise ValueError("path_integral uses the main Dynamics engine, not sampling.engine='native'")
    parallel = config.get("parallel", {})
    if (parallel.get("backend", "serial") != "serial" or parallel.get("mode", "serial") != "serial"
            or parallel.get("bead_workers", 1) != 1 or parallel.get("trajectory_workers", 1) != 1):
        raise ValueError("force-based path_integral workflow currently requires serial execution")
    if config.get("dynamics", {}).get("loop_backend") == "native":
        raise ValueError("sampling action forces require the Python loop (free steps may be native)")
    if method == "suzuki_chin" and table.get("alpha", 0.0) != 0:
        raise ValueError("SC workflow even-bead observables require alpha=0; general alpha forces are available through the Python API")
    if method == "suzuki_chin":
        beta_step = finite_real(table.get("beta_fd_step", 1e-3), "path_integral.beta_fd_step", positive=True)
        if not 8 * np.finfo(float).eps <= beta_step < 1:
            raise ValueError("path_integral.beta_fd_step must be resolvable and strictly below 1")
    output = config.get("output", {})
    if output.get("trajectory_storage", "memory") != "memory" or "chunk_frames" in output:
        raise ValueError("path_integral stores its full saved prefix in memory; use trajectory_storage='memory' without chunk_frames")
    if {"trajectory_stride", "diagnostics", "store_configurations"}.intersection(output):
        raise ValueError("path_integral saves all sampled configurations; trajectory_stride/diagnostics/store_configurations are unsupported")
    if {"parents", "thermostat_params"}.intersection(config.get("sampling", {})):
        raise ValueError("path_integral does not use sampling.parents or thermostat_params")
    if config.get("potential", {}).get("contracted_beads") is not None:
        raise ValueError("path_integral requires independent physical configurations, without a contracted potential")
    options = resolve_materials(config, system)
    options["storage"] = "memory"
    if options["nprod"] < options["interval"]:
        raise ValueError("production must save at least one sample")
    if method in EXCHANGE_METHODS and "block_size" in table:
        if table["block_size"] > options["nprod"] // options["interval"]:
            raise ValueError("path_integral.block_size cannot exceed the number of saved production samples")
    # Construct against a free provider to perform every action validation
    # before the actual potential or output directory is opened.
    from ..potentials.builtin import FreePotential
    _action(config, system, FreePotential())
    return method, options


def _action(config, system, provider):
    table = config["path_integral"]
    method = table.get("method", "suzuki_chin")
    args = (provider, system.masses, config["beta"], config["beads"])
    if method == "suzuki_chin":
        return SuzukiChinPotential(*args, alpha=table.get("alpha", 0.0),
            fd_step=parse_quantity(table.get("fd_step", "1e-4 bohr"), "length"))
    if method in EXCHANGE_METHODS:
        action = BosonicExchangePotential(*args, groups=table.get("groups"), statistics=method,
            cell=system.cell, pbc=system.pbc, periodic_mode=table.get("periodic_mode"))
        for group in action.groups:
            if len({system.symbols[i] for i in group}) != 1:
                raise ValueError("each boson group must contain identical elements")
        return action
    if method != "pigs":
        raise ValueError("unsupported path_integral method")
    trial = table.get("trial", {})
    if (not isinstance(trial, dict) or set(trial) - {"alpha", "center"}
            or "alpha" not in trial):
        raise ValueError("PIGS requires trial.alpha (inverse bohr squared) and optional trial.center (bohr)")
    if "projection_time" not in table:
        raise ValueError("PIGS requires an explicit imaginary projection_time")
    action = PIGSPotential(provider, system.masses, config["beads"],
        parse_quantity(table["projection_time"], "time"), GaussianTrialWavefunction(**trial))
    # Check that the trial broadcasts to the full molecular geometry.
    action.evaluate(np.repeat(system.geometry[None], config["beads"], axis=0))
    return action


def run_path_integral(config, system, output):
    method, options = resolve_path_integral(config, system)
    output = Path(output)
    summaries = []
    for replica in range(options["replicas"]):
        identity = _fingerprint(config, system, {"path_integral_replica": replica})
        checkpoint = output / "checkpoints" / f"path_integral_{replica:04d}.npz"
        physical = make_potential(config["potential"], system)
        try:
            action = _action(config, system, physical)
            dynamics = Dynamics(action, system.masses, action.beta, config["beads"], options["sample_dt"],
                seed=np.random.SeedSequence([int(config.get("seed", 0)), 2219, replica]),
                native=config.get("native", config.get("dynamics", {}).get("native", True)),
                threads=config.get("parallel", {}).get("threads_per_rank", 1),
                **loop_options(config))
            state = State(np.repeat(system.geometry[None], config["beads"], axis=0), dynamics.sample_momenta())
            piston = None
            if options["barostat"] is not None:
                from ..barostat import BZPBarostat
                piston = BZPBarostat(action, system, action.beta, config["beads"], **options["barostat"])
            positions, values, steps, cells, estimator_values = [], [], [], [], []
            elapsed = 0
            total = options["neq"] + options["nprod"]
            if checkpoint.exists():
                with load_checkpoint(checkpoint, expected_kind=KIND, require_integrity=True) as saved:
                    if str(saved["fingerprint"]) != identity:
                        raise ValueError("path-integral checkpoint physics differs")
                    for key in ("elapsed", "step"):
                        scalar = np.asarray(saved[key])
                        if scalar.shape != () or scalar.dtype.kind not in "iu":
                            raise ValueError("path-integral checkpoint clock must contain integer scalars")
                    elapsed = int(saved["elapsed"])
                    if not 0 <= elapsed <= total or int(saved["step"]) != elapsed:
                        raise ValueError("invalid path-integral checkpoint clock")
                    state = State(saved["q"], saved["p"], int(saved["step"]), float(saved["time"]))
                    dynamics.integrator._check(state.q, state.p)
                    if not np.isclose(state.time, elapsed*dynamics.dt, rtol=1e-10, atol=1e-10):
                        raise ValueError("path-integral checkpoint time disagrees with step")
                    positions, values, steps = list(saved["positions"]), list(saved["potential"]), list(saved["sample_steps"])
                    expected = np.arange(options["neq"]+options["interval"], elapsed+1, options["interval"])
                    if (np.asarray(saved["sample_steps"]).dtype.kind not in "iu"
                            or not np.array_equal(steps, expected) or len(values) != len(expected)
                            or np.shape(saved["positions"]) != (len(expected), config["beads"], system.natoms, 3)
                            or not np.isfinite(values).all() or not np.isfinite(positions).all()):
                        raise ValueError("path-integral checkpoint samples disagree with clock/system")
                    if piston is not None:
                        if "barostat" not in saved or "cells" not in saved:
                            raise ValueError("SC-NPT checkpoint is missing the piston/cell history")
                        piston.restore(json.loads(str(saved["barostat"])))
                        saved_cells = np.asarray(saved["cells"])
                        if saved_cells.shape != (len(expected), 3, 3):
                            raise ValueError("SC-NPT checkpoint cell history disagrees with samples")
                        from ..barostat import validate_bzp_snapshot
                        for cell in saved_cells:
                            validate_bzp_snapshot({"method": "bzp", "cell": cell, "momentum": 0.0}, system.cell)
                        cells = list(saved_cells)
                        if len(expected) and expected[-1] == elapsed and not np.array_equal(cells[-1], piston.cell):
                            raise ValueError("SC-NPT last sampled cell differs from the piston")
                    elif "barostat" in saved:
                        raise ValueError("unexpected pressure state in fixed-cell checkpoint")
                    if method == "suzuki_chin" and "sc_estimators" in saved:
                        estimator_values = json.loads(str(saved["sc_estimators"]))
                        required = {"normalization_dof", "internal_energy_hartree", "potential_energy_hartree",
                                    "kinetic_energy_hartree", "scaled_action_second_beta_derivative",
                                    "finite_difference_relative_beta_step"}
                        expected_beta_step = config["path_integral"].get("beta_fd_step", 1e-3) / 2
                        if (not isinstance(estimator_values, list) or len(estimator_values) != len(expected)
                                or any(not isinstance(row, dict) or not required <= set(row) for row in estimator_values)
                                or any(isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value)
                                       for row in estimator_values for value in row.values())
                                or any(set(row) != set(estimator_values[0]) for row in estimator_values)
                                or any(row["normalization_dof"] != 3 * system.natoms
                                       or row["finite_difference_relative_beta_step"] != expected_beta_step
                                       or (piston is not None and "pressure_au" not in row)
                                       for row in estimator_values)):
                            raise ValueError("invalid SC estimator checkpoint history")
                    if method in EXCHANGE_METHODS:
                        if "exchange_estimators" in saved:
                            estimator_values = _exchange_history(
                                json.loads(str(saved["exchange_estimators"])), len(expected), method, system.pbc)
                        elif method == "fermionic" or any(system.pbc):
                            raise ValueError("exchange checkpoint is missing its estimator history")
                    dynamics.rng.bit_generator.state = json.loads(str(saved["rng"]))

            def observe_sc(q):
                volume = piston.volume if piston is not None else (
                    None if system.cell is None else float(abs(np.linalg.det(system.cell))))
                return {key: float(value) for key, value in action.estimators(
                    q, volume=volume,
                    finite_difference_step=config["path_integral"].get("beta_fd_step", 1e-3),
                    require_pressure=piston is not None).items()}

            def observe_exchange(q, physical_result=None):
                volume = float(abs(np.linalg.det(system.cell))) if all(system.pbc) else None
                return {key: float(value) for key, value in action.estimators(
                    q, volume=volume, physical_result=physical_result).items()}

            # Older fixed-cell SC checkpoints contain the full paths. Rebuild
            # their newly available estimators without propagating or drawing
            # noise. An NPT checkpoint may never omit these coupled records.
            if method == "suzuki_chin" and len(estimator_values) != len(positions):
                if piston is not None:
                    raise ValueError("SC-NPT checkpoint is missing its estimator history")
                estimator_values = [observe_sc(q) for q in positions]
            if method in EXCHANGE_METHODS and len(estimator_values) != len(positions):
                # Legacy nonperiodic bosonic checkpoints already store all
                # paths. Reanalysis changes neither identity nor RNG state.
                estimator_values = [observe_exchange(q) for q in positions]

            def save():
                extra = {}
                if method == "suzuki_chin":
                    extra["sc_estimators"] = json.dumps(estimator_values, sort_keys=True, allow_nan=False)
                elif method in EXCHANGE_METHODS:
                    extra["exchange_estimators"] = json.dumps(estimator_values, sort_keys=True, allow_nan=False)
                if piston is not None:
                    extra.update(barostat=json.dumps(piston.snapshot()),
                                 cells=np.asarray(cells).reshape(-1, 3, 3))
                atomic_savez(checkpoint, kind=KIND, fingerprint=identity, elapsed=elapsed,
                    q=state.q, p=state.p, step=state.step, time=state.time,
                    rng=json.dumps(dynamics.rng.bit_generator.state),
                    positions=np.asarray(positions).reshape(-1, config["beads"], system.natoms, 3),
                    potential=np.asarray(values), sample_steps=np.asarray(steps, dtype=np.int64), **extra)

            save()
            while elapsed < total:
                before, rng = state.copy(), copy.deepcopy(dynamics.rng.bit_generator.state)
                before_piston = None if piston is None else piston.snapshot()
                before_stats = copy.deepcopy(dynamics.propagation_stats)
                try:
                    if piston is not None:
                        piston.step(state, dynamics, thermostat=options["thermostat"],
                                    gamma_centroid=options["gamma_centroid"],
                                    andersen_interval=options["andersen_interval"],
                                    method=options["sampling_method"] if options["thermostat"] == "pile-l" else None)
                    elif options["thermostat"] == "pile-l":
                        dynamics.sample_step(state, gamma_centroid=options["gamma_centroid"],
                                             method=options["sampling_method"])
                    else:
                        if state.step % options["andersen_interval"] == 0:
                            state.p[:] = dynamics.sample_momenta()
                        dynamics.step(state)
                    record = state.step > options["neq"] and (state.step-options["neq"]) % options["interval"] == 0
                    if record:
                        if method == "pigs":
                            observed = state.q[len(state.q)//2:len(state.q)//2+1]
                        elif method == "suzuki_chin":
                            observed = state.q[::2]
                        else:
                            observed = state.q
                        physical_result = validate_result(physical.evaluate(observed), observed.shape)
                        potential = float(physical_result.energy.mean())
                        if method == "suzuki_chin":
                            estimators = observe_sc(state.q)
                        elif method in EXCHANGE_METHODS:
                            estimators = observe_exchange(state.q, physical_result)
                            _exchange_history([estimators], 1, method, system.pbc)
                            if estimator_values and set(estimators) != set(estimator_values[0]):
                                raise ValueError("exchange estimator columns changed during sampling; "
                                                 "physical virial availability must remain consistent")
                except BaseException:
                    state = before
                    dynamics.rng.bit_generator.state = rng
                    dynamics.propagation_stats = before_stats
                    if piston is not None:
                        piston.restore(before_piston)
                    save()
                    raise
                elapsed = state.step
                if record:
                    positions.append(state.q.copy())
                    values.append(potential)
                    steps.append(state.step)
                    if method == "suzuki_chin" or method in EXCHANGE_METHODS:
                        estimator_values.append(estimators)
                    if piston is not None:
                        cells.append(piston.cell)
                if elapsed % options["checkpoint_steps"] == 0:
                    save()
            save()
            summary = {"replica": replica, "samples": len(values),
                "potential_energy_hartree": None if not np.isfinite(values).all() else float(np.mean(values)),
                "checkpoint": str(checkpoint)}
            if method == "suzuki_chin":
                summary["estimators"] = {key: float(np.mean([row[key] for row in estimator_values]))
                                         for key in estimator_values[0]}
                if len(estimator_values) >= 2:
                    from ..suzuki_chin import suzuki_chin_thermodynamics
                    series = {key: np.array([row[key] for row in estimator_values])
                              for key in estimator_values[0]}
                    kwargs = {} if piston is None else {
                        "volumes": np.abs(np.linalg.det(cells)), "pressure_au": piston.pressure_au}
                    summary["thermodynamics"] = suzuki_chin_thermodynamics(series, action.beta, **kwargs)
                if piston is not None:
                    summary["barostat"] = piston.snapshot()
                    summary["mean_volume_bohr3"] = float(np.mean(np.abs(np.linalg.det(cells))))
            elif method in EXCHANGE_METHODS:
                from ..quantum_statistics import exchange_thermodynamics
                summary["thermodynamics"] = exchange_thermodynamics(
                    estimator_values, **_exchange_statistics_options(config["path_integral"]))
                if method == "fermionic":
                    summary["bosonic_sampling_potential_energy_hartree"] = summary["potential_energy_hartree"]
                    summary["potential_energy_hartree"] = summary["thermodynamics"]["potential_energy_hartree"]
            summaries.append(summary)
        finally:
            physical.close()
    return {"task": "pimd", "method": method, "equilibrium_only": True,
        "trajectory_storage": "memory", "status": "complete",
        "replicas": summaries, "samples": sum(x["samples"] for x in summaries),
        "observable": ("projected_midpoint_potential" if method == "pigs" else
            "even_bead_potential_alpha_zero_only" if method == "suzuki_chin" else
            "fermionic_sign_reweighted_thermodynamics" if method == "fermionic" else "symmetric_bead_potential"),
        "temperature_K": None if method == "pigs" else config["temperature_K"],
        "projection_time_au": action.projection_time if method == "pigs" else None,
        "ensemble": "npt" if options["barostat"] is not None else "fixed_cell",
        **({"physical_converged": False,
            "sampling_statistics": "bosonic",
            "periodic_exchange": config["path_integral"].get("periodic_mode"),
            "warnings": [dict(warning, replica=row["replica"]) for row in summaries
                         for warning in row["thermodynamics"]["warnings"]]}
           if method in EXCHANGE_METHODS else {}),
        "limitations": (["SC thermodynamic estimates include explicit temperature derivatives of the force-squared action",
            "NPT heat capacity is Cp; fixed-volume heat capacity is Cv",
            "sampling, time-step, finite-difference and bead convergence must be established separately",
            "equilibrium distinguishable-nucleus statistics; no real-time dynamics"]
            if method == "suzuki_chin" else
            ["symmetric equilibrium observables only; no real-time dynamics or heat capacity",
             "fermionic ratios require a resolved positive mean sign; unresolved conclusions are withheld",
             "block errors are conditional; sampling, time-step and bead convergence remain unestablished",
             "periodic_mode='minimum_image' is an explicit approximation without a winding sum",
             "centroid BZP NPT is rejected because its pressure is not invariant under independent bead images"]
            if method in EXCHANGE_METHODS else
            ["potential energy only; not a total-energy or heat-capacity estimator",
             "sampling, time-step and bead/projection convergence must be established separately",
             "no real-time, pressure or permutation-resolved particle estimator"])}
