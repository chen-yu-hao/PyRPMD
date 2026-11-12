"""Read advisory checkpoint clocks without loading trajectories or calling a PES.

These readers do not replace checkpoint integrity validation on restart. Their
only consumers are the run progress reporter and its tests. Unknown budgets or
unreadable clocks stay unknown rather than being treated as completed work.
"""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np


_MAX_METADATA = 1024 * 1024


def _count(value):
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= 0:
        return int(value)
    raise ValueError("checkpoint clock must be a nonnegative integer")


def _json(path):
    with path.open("rb") as source:
        raw = source.read(8 * _MAX_METADATA + 1)
    if len(raw) > 8 * _MAX_METADATA:
        raise ValueError("progress metadata exceeds its size limit")
    return json.loads(raw)


def _clock(path, *, paired=False):
    # Open once: an atomic replacement must not mix two checkpoint generations.
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())

        def scalar(name):
            member = name + ".npy"
            if member not in names:
                return None
            if archive.getinfo(member).file_size > _MAX_METADATA:
                raise ValueError("checkpoint clock metadata is too large")
            a = np.load(BytesIO(archive.read(member)), allow_pickle=False)
            if a.shape != ():
                raise ValueError("checkpoint clock metadata must be scalar")
            return a.item()

        if paired:
            # A child checkpoint is committed only after BOTH signed children.
            # Read the curve's NPY header, never its potentially large values.
            with archive.open("numerator.npy") as source:
                version = np.lib.format.read_magic(source)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(source)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(source)
                else:
                    raise ValueError("unsupported child clock array version")
            if len(shape) != 1 or dtype.hasobject:
                raise ValueError("invalid child checkpoint curve")
            return 2 * _count(shape[0])
        for name in ("elapsed", "step"):
            value = scalar(name)
            if value is not None:
                return _count(value)
        for name in ("metadata", "meta"):
            value = scalar(name)
            if value is not None:
                metadata = json.loads(value)
                for key in ("done", "elapsed", "step"):
                    if key in metadata:
                        return _count(metadata[key])
        raise ValueError("checkpoint has no supported progress clock")


def _adaptive(value):
    if isinstance(value, dict):
        return any(("adaptive" in key and bool(item)
                    and not (isinstance(item, dict) and item.get("enabled") is False))
                   or ("adaptive" not in key and _adaptive(item))
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_adaptive(item) for item in value)
    return False


class CheckpointProgress:
    """Aggregate only recognized committed clocks, once per changed file.

    MPI ranks and local workers need no reporter, locks, queues or stdout calls:
    the coordinator observes their existing atomic checkpoints on shared storage.
    """

    def __init__(self, directory, config, plan=(), *, checkpoint_directory=None):
        self.directory = Path(directory)
        self.config = config
        self.checkpoint_directory = Path(checkpoint_directory or directory)
        self.plan = list(plan)
        self._cache = {}
        self.warnings = []

    def _read(self, path, *, paired=False, journal=False):
        try:
            stat = path.stat()
            stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns, paired, journal)
            cached = self._cache.get(path)
            if cached is not None and cached[0] == stamp:
                return cached[1]
            if journal:
                data = _json(path)
                value = _count(data["blocks"][-1]["end"])
            else:
                value = _clock(path, paired=paired)
            self._cache[path] = (stamp, value)
            return value
        except (OSError, ValueError, TypeError, KeyError, IndexError, EOFError, zipfile.BadZipFile) as error:
            self.warnings.append(f"Cannot read progress clock {path.relative_to(self.directory)}: {error}")
            return None

    def _sum(self, pattern, *, paired=False, journal=False, multiplier=1, directory=None):
        values = [self._read(path, paired=paired, journal=journal)
                  for path in sorted((directory or self.directory).glob(pattern))]
        if any(value is None for value in values):
            return None
        return sum(values) * multiplier

    def _budget(self, stage):
        entries = [entry for entry in self.plan if entry.get("stage") == stage]
        if not entries or any("unavailable" in str(e.get("reason", "")) for e in entries):
            return None
        try:
            value = sum(_count(entry["planned_steps"]) for entry in entries)
            return value if value > 0 else None
        except (KeyError, ValueError):
            return None

    def snapshot(self):
        self.warnings = []
        task = self.config.get("task", "unknown")
        sampling = self.config.get("sampling", {})
        dynamic = _adaptive({key: self.config.get(key) for key in ("umbrella", "adaptive_umbrella", "rate")})
        stages = []
        scope = "planned integration steps; excludes setup and analysis"

        def add(name, done, total):
            if dynamic:
                total = None
            if done is not None and total is not None and done > total:
                self.warnings.append(f"{name} clock exceeds the declared budget; percentage withheld")
                total = None
            stages.append({"stage": name, "completed_steps": done, "total_steps": total})

        if "tempering" in self.config:
            add("tempering", None, None)
        elif task in ("pimd", "diffusion") and "path_integral" not in self.config:
            neq = _count(sampling.get("equilibration_steps", 0))
            nprod = _count(sampling.get("production_steps", 0))
            replicas = _count(sampling.get("replicas", 1))
            engine = sampling.get("engine", "python")
            if engine == "native":
                paths = sorted(self.directory.glob("samples/native_pimd_*/journal.json"))
                clocks = [self._read(path, journal=True) for path in paths]
            elif engine == "torch":
                paths = sorted(self.directory.glob("checkpoints/materials_torch.npz"))
                clocks = [self._read(path) for path in paths] * replicas
            else:
                paths = sorted(self.directory.glob("checkpoints/materials_*.npz"))
                clocks = [self._read(path) for path in paths]
            bad = any(value is None for value in clocks)
            add("equilibration", None if bad else sum(min(value, neq) for value in clocks), neq * replicas)
            add("production", None if bad else sum(max(0, value - neq) for value in clocks), nprod * replicas)
        elif task in ("fes", "rate"):
            method = self.config.get("fes", {}).get("method", self.config.get("method", "umbrella"))
            if task == "rate":
                stage = self.config.get("rate", {}).get("pmf_stage", {})
                method = stage.get("sampler", "umbrella")
                if self.config.get("rate", {}).get("pmf_source") not in (None, "umbrella"):
                    method = "external"
            if method == "external":
                pass
            elif dynamic:
                add("adaptive umbrella", None, None)
            elif self.config.get("replica_exchange", {}).get("enabled"):
                add("replica exchange", None, self._budget("umbrella"))
            elif method == "umbrella":
                prefix = "umbrella/" if task == "rate" else ""
                add("umbrella", self._sum(prefix + "checkpoints/window_*.npz"), self._budget("umbrella"))
            elif method in ("metadynamics", "opes"):
                prefix = "pmf/" if task == "rate" else ""
                source_config = self.config.get("rate", {}).get("pmf_stage", {}).get("config", self.config) if task == "rate" else self.config
                shared = source_config.get(method, {})
                if any(shared.get(key) for key in ("shared_walkers", "multiple_walkers", "distributed")):
                    add(method, None, None)
                else:
                    sample = source_config.get("sampling", {})
                    budget = (_count(sample.get("equilibration_steps", 0)) + _count(sample.get("production_steps", 0))) * _count(sample.get("replicas", 1))
                    add(method, self._sum(prefix + f"checkpoints/{method}_[0-9]*.npz"), budget or None)
            else:
                add(method, None, None)
            if task == "rate":
                add("recrossing parent", self._sum("checkpoints/parent_*.npz", directory=self.checkpoint_directory), self._budget("recrossing parent"))
                add("recrossing child", self._sum("checkpoints/child_*.npz", paired=True, directory=self.checkpoint_directory), self._budget("recrossing child"))
        elif task == "drpmd":
            collision = self.config.get("collision", {})
            budget = collision.get("max_steps")
            ledger = self.directory / "results" / "extensions.json"
            if ledger.is_file():
                try:
                    saved = _json(ledger)
                    if saved.get("format") != "PyRPMD.drpmd.extensions.v1":
                        raise ValueError("unknown dRPMD extension ledger")
                    budget = _count(saved["requests"][-1]["max_steps"])
                except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
                    budget = None
                    self.warnings.append(f"Cannot read authorized collision budget: {error}")
            number = collision.get("trajectories")
            total = _count(budget) * _count(number) if budget is not None and number is not None else None
            add("collisions", self._sum("trajectories/trajectory-*.checkpoint.npz"), total)
            scope = "collision propagation budget only; excludes parent preparation; early stopping may use less than 100%"
        else:
            add(task, None, None)
            scope = "this workflow does not expose a supported integration-step budget"

        known_done = all(stage["completed_steps"] is not None for stage in stages)
        known_total = all(stage["total_steps"] is not None for stage in stages)
        completed = sum(stage["completed_steps"] for stage in stages) if known_done else None
        total = sum(stage["total_steps"] for stage in stages) if known_total else None
        percent = 100.0 * completed / total if completed is not None and total else None
        active = [stage for stage in stages if stage["completed_steps"] is None
                  or stage["total_steps"] is None or stage["completed_steps"] < stage["total_steps"]]
        # Later-stage checkpoints are stronger evidence than an incomplete or
        # skipped earlier stage (e.g. an externally supplied PMF).
        observed = [stage for stage in stages if stage["completed_steps"]]
        current = active[0]["stage"] if active else "analysis"
        if observed:
            latest = observed[-1]
            if latest["total_steps"] is None or latest["completed_steps"] < latest["total_steps"]:
                current = latest["stage"]
        return {"stage": current, "completed_steps": completed, "total_steps": total,
                "percent": percent, "stages": stages, "scope": scope,
                "warnings": self.warnings.copy()}
