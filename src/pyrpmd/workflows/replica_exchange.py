"""Same-temperature fixed-window REUS with synchronized ensemble checkpoints.

Each task owns an independent ladder. Swaps move q/p; RNGs and clocks stay at
windows. Samples are window-labelled and precede coincident exchanges. A whole
propagation/exchange event, not an individual window, is the rollback unit.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..bias import UmbrellaBias, displacement
from ..checkpoint import atomic_savez, load_checkpoint
from ..dynamics import Dynamics, State
from ..parallel import make_task_potential, root_call, run_tasks
from ..propagation import loop_options
from .fes import _analyze_streams, resolve_umbrellas, steer_to_center
from .common import _fingerprint, checked_integer, write_json, fingerprint_matches
from .sampling import checkpoint_cv_config, fes_sampler

CHECKPOINT_KIND = "PyRPMD.replica_exchange.v1"
_FIELDS = {"enabled", "interval_steps", "pairs"}


def _pairs(centers, stiffness, beta, periods, explicit=None):
    """Explicit connected graph, or deterministic metric minimum spanning tree."""
    n = len(centers)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if explicit is None:
        if n > 2048:
            raise ValueError("provide replica_exchange.pairs for more than 2048 windows")
        distances = []
        for i in range(n):
            delta = displacement(centers[i + 1:], centers[i], periods)
            metric = beta * np.einsum("ni,ij,nj->n", delta, stiffness, delta)
            if not np.isfinite(metric).all():
                raise ValueError("nonfinite exchange-window distances")
            distances.extend((float(d), i, j) for j, d in enumerate(metric, i + 1))
        pairs = []
        for _, i, j in sorted(distances):
            a, b = find(i), find(j)
            if a != b:
                parent[a] = b
                pairs.append((i, j))
                if len(pairs) == n - 1:
                    break
    else:
        raw = np.asarray(explicit)
        if (raw.ndim != 2 or raw.shape[1:] != (2,) or not len(raw)
                or raw.dtype.kind not in "iu" or np.any(raw < 0) or np.any(raw >= n)):
            raise ValueError("replica_exchange.pairs must contain in-range integer window pairs")
        pairs = [tuple(sorted(map(int, pair))) for pair in raw]
        if any(i == j for i, j in pairs) or len(set(pairs)) != len(pairs):
            raise ValueError("exchange pairs must be distinct, without self-pairs")
        pairs.sort()
        for i, j in pairs:
            parent[find(i)] = find(j)
    if len({find(i) for i in range(n)}) != 1:
        raise ValueError("replica_exchange.pairs must connect every window")
    return np.asarray(pairs, dtype=int)


def exchange_options(config, system):
    """Validate before creating output or potential resources."""
    table = config.get("replica_exchange", {})
    if not isinstance(table, Mapping) or set(table) - _FIELDS:
        raise ValueError("replica_exchange accepts enabled, interval_steps and pairs only")
    enabled = table.get("enabled", bool(table))
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("replica_exchange.enabled must be boolean")
    if not enabled:
        return {"enabled": False}
    checkpoint_cv_config(config)  # fail before output/resources for unsafe object identities
    if config.get("task", "fes") != "fes":
        raise ValueError("replica exchange currently requires task='fes'")
    if config.get("analysis", {}).get("measure", "cv_density") != "cv_density":
        raise ValueError("replica exchange currently requires the cv_density measure")
    if config.get("adaptive_umbrella", {}).get("enabled", bool(config.get("adaptive_umbrella"))):
        raise ValueError("replica exchange requires fixed windows; adaptive_umbrella is not combined")
    sampler = fes_sampler(config)
    cv, specs, centers, stiffness = resolve_umbrellas(config, system)
    if len(centers) < 2:
        raise ValueError("replica exchange requires at least two umbrella windows")
    sampling = config["sampling"]
    neq = checked_integer(sampling["equilibration_steps"], "sampling.equilibration_steps", 0)
    nprod = checked_integer(sampling["production_steps"], "sampling.production_steps")
    interval = checked_integer(sampling.get("sample_interval", sampling.get("interval_steps", 1)),
                               "sampling.sample_interval")
    if nprod // interval < 2:
        raise ValueError("replica exchange requires at least two saved samples per window")
    dt = sampling.get("dt_au", config["dt_au"])
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(config["beta"]) or config["beta"] <= 0:
        raise ValueError("replica exchange beta and time step must be positive and finite")
    checked_integer(config["beads"], "beads")
    checked_integer(config.get("seed", 0), "seed", 0)
    return {
        "enabled": True, "cv": cv, "specs": specs, "centers": centers,
        "stiffness": stiffness, "sampler": sampler, "dt": dt,
        "neq": neq, "nprod": nprod, "sample_interval": interval,
        "andersen_interval": checked_integer(sampling.get("andersen_interval", 10), "andersen_interval"),
        "replicas": checked_integer(sampling.get("replicas", 1), "sampling.replicas"),
        "checkpoint_steps": checked_integer(config.get("output", {}).get("checkpoint_steps", 1000),
                                            "output.checkpoint_steps"),
        "interval_steps": checked_integer(table.get("interval_steps", 100), "replica_exchange.interval_steps"),
        "pairs": _pairs(centers, stiffness, config["beta"], cv.periods, table.get("pairs")),
    }


def exchange_log_ratio(bias_i, bias_j, values_i, values_j, beta):
    """Log target ratio for whole-ring q/p swaps at common T/P/masses/PES.

    H_P contains P*W and is sampled at beta/P: the exponent is -beta*Delta(W),
    with no additional P. Spring, kinetic and physical potential terms cancel.
    """
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("exchange beta must be positive and finite")
    energies = np.asarray([bias_i.physical_energy(values_j), bias_j.physical_energy(values_i),
                           bias_i.physical_energy(values_i), bias_j.physical_energy(values_j)])
    if energies.shape != (4,) or not np.isfinite(energies).all():
        raise ValueError("exchange requires finite scalar physical umbrella energies")
    e = energies.astype(np.longdouble)
    with np.errstate(over="ignore"):
        result = float(-np.longdouble(beta) * ((e[0] - e[2]) + (e[1] - e[3])))
    if np.isnan(result):
        raise ValueError("indeterminate exchange log ratio")
    return result


def _exchange_sweep(states, biases, cv, rng, beta, pairs, labels, attempts, accepts):
    values = np.asarray([cv.value(state.q) for state in states])
    if values.shape != (len(states), cv.dimension) or not np.isfinite(values).all():
        raise ValueError("CV returned invalid exchange observations")
    decisions = np.zeros(len(pairs), dtype=bool)
    for edge in rng.permutation(len(pairs)):
        i, j = pairs[edge]
        log_ratio = exchange_log_ratio(biases[i], biases[j], values[i], values[j], beta)
        uniform = rng.random()  # one draw even for certain acceptance
        log_uniform = -np.inf if uniform == 0 else np.log(uniform)
        attempts[edge] += 1
        if log_uniform < min(0.0, log_ratio):
            q, p = states[i].q.copy(), states[i].p.copy()
            states[i].q[:], states[i].p[:] = states[j].q, states[j].p
            states[j].q[:], states[j].p[:] = q, p
            values[[i, j]] = values[[j, i]]
            labels[[i, j]] = labels[[j, i]]
            accepts[edge] += 1
            decisions[edge] = True
    return decisions


def _path(output, replica):
    return Path(output) / "checkpoints" / f"exchange_{replica:04d}.npz"


def _identity(config, system, options, replica):
    return _fingerprint(checkpoint_cv_config(config), system, {"kind": CHECKPOINT_KIND, "replica": replica,
                       "centers": options["centers"], "stiffness": options["stiffness"],
                       "pairs": options["pairs"]})


def _read(config, system, options, output, replica, *, complete=False):
    path = _path(output, replica)
    with load_checkpoint(path, expected_kind=CHECKPOINT_KIND, require_integrity=True) as archive:
        data = {name: archive[name] for name in archive.files}
    identity = {"kind": CHECKPOINT_KIND, "replica": replica,
                "centers": options["centers"], "stiffness": options["stiffness"],
                "pairs": options["pairs"]}
    if not fingerprint_matches(data["fingerprint"], checkpoint_cv_config(config), system, identity):
        raise ValueError(f"checkpoint physics differs: {path}")
    if (data["payload_version"].shape != () or data["payload_version"].dtype.kind not in "iu"
            or data["payload_version"] != 1):
        raise ValueError("unsupported replica-exchange payload version")
    elapsed = data["elapsed"]
    total = options["neq"] + options["nprod"]
    if elapsed.shape != () or elapsed.dtype.kind not in "iu" or not 0 <= elapsed <= total:
        raise ValueError("invalid exchange checkpoint clock")
    elapsed = int(elapsed)
    if complete and elapsed != total:
        raise ValueError("offline exchange analysis requires complete ladder checkpoints")
    nwin, dim = len(options["centers"]), options["cv"].dimension
    shape = (nwin, config["beads"], len(system.masses), 3)
    nobs = max(0, elapsed - options["neq"]) // options["sample_interval"]
    nsweep = elapsed // options["interval_steps"]
    npair = len(options["pairs"])
    for key, expected in (("q", shape), ("p", shape), ("times", (nwin,)),
                          ("samples", (nwin, nobs, dim)), ("energies", (nwin, nobs))):
        value = data[key]
        if value.shape != expected or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"invalid exchange checkpoint {key}")
    if (not np.all(data["times"] == data["times"][0]) or
            not np.allclose(data["times"], elapsed * options["dt"], rtol=1e-10, atol=options["dt"] * 1e-8)):
        raise ValueError("exchange checkpoint windows have inconsistent clocks")
    fixed = np.asarray(getattr(system, "fixed_atoms", ()), dtype=int)
    if len(fixed) and (np.any(data["q"][:, :, fixed, :] != system.geometry[fixed]) or
                       np.any(data["p"][:, :, fixed, :] != 0)):
        raise ValueError("exchange checkpoint moved fixed atoms")
    expected_steps = options["neq"] + np.arange(1, nobs + 1) * options["sample_interval"]
    if data["sample_steps"].dtype.kind not in "iu" or not np.array_equal(data["sample_steps"], expected_steps):
        raise ValueError("exchange checkpoint sample clock mismatch")
    if not np.array_equal(data["pairs"], options["pairs"]):
        raise ValueError("exchange checkpoint graph mismatch")
    history, walkers = data["walker_history"], data["walker_ids"]
    if (history.shape != (nsweep + 1, nwin) or history.dtype.kind not in "iu" or
            not np.all(np.sort(history, axis=1) == np.arange(nwin)) or
            not np.array_equal(history[0], np.arange(nwin))):
        raise ValueError("invalid exchange walker permutation history")
    expected_walkers = history[(expected_steps - 1) // options["interval_steps"]].T
    if walkers.dtype.kind not in "iu" or not np.array_equal(walkers, expected_walkers):
        raise ValueError("exchange samples have inconsistent walker identities")
    attempts, accepts = data["attempts"], data["accepts"]
    decisions = data["decisions"]
    if (attempts.shape != (npair,) or accepts.shape != (npair,) or
            attempts.dtype.kind not in "iu" or accepts.dtype.kind not in "iu" or
            np.any(attempts != nsweep) or np.any(accepts < 0) or np.any(accepts > attempts) or
            decisions.shape != (nsweep, npair) or decisions.dtype.kind != "b" or
            not np.array_equal(decisions.sum(axis=0), accepts)):
        raise ValueError("invalid exchange acceptance counters")
    rngs = json.loads(str(data["rngs"]))
    stats = json.loads(str(data["propagation"]))
    if not isinstance(rngs, list) or len(rngs) != nwin or not isinstance(stats, list) or len(stats) != nwin:
        raise ValueError("invalid exchange per-window metadata")
    for state in [*rngs, json.loads(str(data["exchange_rng"]))]:
        generator = np.random.default_rng()
        generator.bit_generator.state = state
    return data


def _advance_window(dynamics, state, bias, cv, stop, options, values, energies, walkers, label):
    sampler = options["sampler"]
    while state.step < stop:
        if sampler.thermostat == "andersen":
            interval = options["andersen_interval"]
            if state.step % interval == 0:
                state.p[:] = dynamics.sample_momenta()
            count = min(stop - state.step, dynamics.block_steps, 4096,
                        interval - state.step % interval)
            production = state.step + np.arange(1, count + 1) - options["neq"]
            observe = (production > 0) & (production % options["sample_interval"] == 0)
            block = dynamics.propagate_block(state, count, bias=bias, cv=cv,
                                            observe=observe, check_energy=True)
            values.extend(block.values.copy())
            energies.extend(block.energies.tolist())
            walkers.extend([int(label)] * len(block.values))
        else:
            sampler.step(dynamics, state, bias)
            production = state.step - options["neq"]
            if production > 0 and production % options["sample_interval"] == 0:
                value = np.asarray(cv.value(state.q), dtype=float)
                energy = float(dynamics.energy(state, bias))
                if value.shape != (cv.dimension,) or not np.isfinite(value).all() or not np.isfinite(energy):
                    raise ValueError("nonfinite or invalid exchange sample")
                values.append(value.copy())
                energies.append(energy)
                walkers.append(int(label))
            dynamics._record_python_steps(1)


def _sample_ladder(task):
    config, system, replica, output = (task[k] for k in ("config", "system", "replica", "output"))
    options = exchange_options(config, system)
    path = _path(output, replica)
    saved = _read(config, system, options, output, replica) if path.exists() else None
    total = options["neq"] + options["nprod"]
    if saved is not None and int(saved["elapsed"]) == total:
        return {"replica": replica, "data": saved, "checkpoint": str(path)}
    cv, centers, stiffness = (options[k] for k in ("cv", "centers", "stiffness"))
    nwin = len(centers)
    provider = make_task_potential(config["potential"], system)
    try:
        dynamics, states, biases = [], [], []
        for i, center in enumerate(centers):
            dyn = Dynamics(provider, system.masses, config["beta"], config["beads"], options["dt"],
                           seed=np.random.SeedSequence([config.get("seed", 0), 802, replica, i]),
                           native=config.get("native", config.get("dynamics", {}).get("native", True)),
                           threads=config.get("parallel", {}).get("threads_per_rank", 1),
                           fixed_atoms=getattr(system, "fixed_atoms", ()), fixed_positions=system.geometry,
                           **loop_options(config))
            if saved is None:
                q = np.repeat(system.geometry[None], config["beads"], axis=0)
                q = steer_to_center(q, cv, center, system.masses, fixed_atoms=dyn.fixed_atoms)
                state = State(q, dyn.sample_momenta())
            else:
                state = State(saved["q"][i].copy(), saved["p"][i].copy(), int(saved["elapsed"]),
                              float(saved["times"][i]))
                dyn.rng.bit_generator.state = json.loads(str(saved["rngs"]))[i]
                dyn.propagation_stats = json.loads(str(saved["propagation"]))[i]
            dynamics.append(dyn)
            states.append(state)
            biases.append(UmbrellaBias(cv, center, stiffness))
        rng = np.random.default_rng(np.random.SeedSequence([config.get("seed", 0), 801, replica]))
        values, energies, walkers = ([[] for _ in centers] for _ in range(3))
        labels = np.arange(nwin)
        attempts = np.zeros(len(options["pairs"]), dtype=int)
        accepts = attempts.copy()
        decisions, history = [], [labels.copy()]
        elapsed = 0
        if saved is not None:
            values = [list(a) for a in saved["samples"]]
            energies = [list(a) for a in saved["energies"]]
            walkers = [list(a) for a in saved["walker_ids"]]
            attempts[:], accepts[:] = saved["attempts"], saved["accepts"]
            decisions, history = list(saved["decisions"]), list(saved["walker_history"])
            labels[:] = history[-1]
            rng.bit_generator.state = json.loads(str(saved["exchange_rng"]))
            elapsed = int(saved["elapsed"])
        saved_elapsed = None if saved is None else elapsed

        def payload():
            nobs = len(values[0])
            return dict(payload_version=np.asarray(1), elapsed=np.asarray(elapsed),
                        fingerprint=np.asarray(_identity(config, system, options, replica)),
                        q=np.stack([s.q for s in states]), p=np.stack([s.p for s in states]),
                        times=np.array([s.time for s in states]),
                        samples=np.asarray(values, dtype=float).reshape(nwin, nobs, cv.dimension),
                        energies=np.asarray(energies, dtype=float).reshape(nwin, nobs),
                        walker_ids=np.asarray(walkers, dtype=int).reshape(nwin, nobs),
                        sample_steps=options["neq"] + np.arange(1, nobs + 1) * options["sample_interval"],
                        walker_history=np.asarray(history, dtype=int),
                        pairs=options["pairs"], attempts=attempts, accepts=accepts,
                        decisions=np.asarray(decisions, dtype=bool).reshape(-1, len(attempts)),
                        rngs=np.asarray(json.dumps([d.rng.bit_generator.state for d in dynamics])),
                        exchange_rng=np.asarray(json.dumps(rng.bit_generator.state)),
                        propagation=np.asarray(json.dumps([d.propagation_stats for d in dynamics])))

        def save():
            nonlocal saved_elapsed
            if saved_elapsed != elapsed:
                atomic_savez(path, kind=CHECKPOINT_KIND, **payload())
                saved_elapsed = elapsed

        save()
        while elapsed < total:
            stop = min(total, elapsed + options["interval_steps"] - elapsed % options["interval_steps"],
                       elapsed + options["checkpoint_steps"] - elapsed % options["checkpoint_steps"])
            before = ([s.copy() for s in states], [copy.deepcopy(d.rng.bit_generator.state) for d in dynamics],
                      [copy.deepcopy(d.propagation_stats) for d in dynamics], copy.deepcopy(rng.bit_generator.state),
                      labels.copy(), attempts.copy(), accepts.copy(), len(values[0]), len(history))
            try:
                for i, (dyn, state, bias) in enumerate(zip(dynamics, states, biases)):
                    _advance_window(dyn, state, bias, cv, stop, options,
                                    values[i], energies[i], walkers[i], labels[i])
                if any(s.step != stop or s.time != states[0].time for s in states):
                    raise RuntimeError("exchange windows did not reach a common clock")
                if stop % options["interval_steps"] == 0:
                    decisions.append(_exchange_sweep(states, biases, cv, rng, config["beta"],
                                                      options["pairs"], labels, attempts, accepts))
                    history.append(labels.copy())
            except BaseException:
                # A half-propagated window or half-sweep is not a valid ensemble.
                old_states, old_rngs, old_stats, old_exchange, old_labels, old_attempts, old_accepts, nobs, nhist = before
                for i, (dyn, state, old) in enumerate(zip(dynamics, states, old_states)):
                    state.q[:], state.p[:] = old.q, old.p
                    state.step, state.time = old.step, old.time
                    dyn.rng.bit_generator.state = old_rngs[i]
                    dyn.propagation_stats = old_stats[i]
                    del values[i][nobs:]
                    del energies[i][nobs:]
                    del walkers[i][nobs:]
                rng.bit_generator.state = old_exchange
                labels[:], attempts[:], accepts[:] = old_labels, old_attempts, old_accepts
                del history[nhist:]
                del decisions[nhist - 1:]
                save()
                raise
            elapsed = stop
            if elapsed % options["checkpoint_steps"] == 0 or elapsed == total:
                # Atomic write failure leaves either old or new complete ensemble.
                save()
        return {"replica": replica, "data": payload(), "checkpoint": str(path)}
    finally:
        if hasattr(provider, "close"):
            provider.close()


def _analyze(ladders, config, system, output, return_data):
    # External executor.map returns a one-shot iterator. Every window and the
    # exchange diagnostics must visit the same complete set of ladders.
    ladders = list(ladders)
    if not ladders:
        raise ValueError("replica-exchange analysis requires at least one ladder")
    options = exchange_options(config, system)
    streams = []
    for window, center in enumerate(options["centers"]):
        for ladder in ladders:
            replica, data = ladder["replica"], ladder["data"]
            streams.append(dict(window=window, replica=replica, exchange_group=replica,
                                center=center, stiffness=options["stiffness"],
                                samples=data["samples"][window], energies=data["energies"][window],
                                walker_ids=data["walker_ids"][window], last_q=data["q"][window],
                                last_p=data["p"][window], checkpoint=ladder["checkpoint"],
                                propagation=json.loads(str(data["propagation"]))[window]))
    result = _analyze_streams(streams, config, system, output, return_data)
    summary = result["summary"] if return_data else result
    diagnostics = []
    for ladder in ladders:
        data = ladder["data"]
        visits = np.zeros((len(options["centers"]), len(options["centers"])), dtype=int)
        for labels in data["walker_history"]:
            visits[labels, np.arange(len(labels))] += 1
        diagnostics.append({"replica": ladder["replica"], "attempts": data["attempts"],
                            "accepts": data["accepts"], "acceptance_fraction": np.divide(
                                data["accepts"], data["attempts"],
                                out=np.full(len(data["attempts"]), np.nan), where=data["attempts"] > 0),
                            "walker_window_visits": visits})
    exchange = {"method": "same_temperature_replica_exchange_umbrella",
                "independent_ladders": len(ladders), "windows_per_ladder": len(options["centers"]),
                "pairs": options["pairs"], "interval_steps": options["interval_steps"],
                "sampler": vars(options["sampler"]), "ladders": diagnostics,
                "sample_convention": "window-labelled, before coincident exchanges",
                "uncertainty": "synchronous circular blocks across windows within each ladder",
                "parallelism": "independent ladders; windows inside a ladder are propagated sequentially",
                "physical_converged": False}
    write_json(Path(output) / "exchange.json", exchange)
    summary["replica_exchange"] = exchange
    summary["warnings"].append("Exchange acceptance and walker visits do not prove equilibration or hidden-coordinate mixing; exchanged trajectory time is not physical RPMD time.")
    write_json(Path(output) / "summary.json", summary)
    return result


def run_replica_exchange_fes(config, system, output, executor=None, return_data=False):
    """Run independent same-temperature ladders and reconstruct the unbiased FES."""
    options = exchange_options(config, system)
    if not options["enabled"]:
        raise ValueError("replica_exchange must be enabled")
    tasks = [dict(config=config, system=system, replica=r, output=str(output))
             for r in range(options["replicas"])]
    ladders = (executor.map(_sample_ladder, tasks) if executor is not None else
               run_tasks(_sample_ladder, tasks, config.get("parallel", {})))
    return root_call(_analyze, ladders, config, system, str(output), return_data)


def reanalyze_replica_exchange_fes(config, system, output, return_data=False):
    """Read complete atomic ladders without constructing a PES or propagating."""
    def analyze():
        options = exchange_options(config, system)
        ladders = [dict(replica=r, data=_read(config, system, options, output, r, complete=True),
                        checkpoint=str(_path(output, r))) for r in range(options["replicas"])]
        return _analyze(ladders, config, system, str(output), return_data)
    return root_call(analyze)
