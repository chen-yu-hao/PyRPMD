"""Versioned synchronous host-memory C ABI and cached object-file linking."""

import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import threading
import tomllib
import numpy as np
from .base import Potential, PESResult, PotentialError, coordinates, validate_result

I32 = C.c_int32
I64 = C.c_int64
DP = C.POINTER(C.c_double)


class Error(C.Structure):
    _fields_ = [("message", C.c_void_p), ("capacity", I64), ("required_capacity", I64)]


class Options(C.Structure):
    _fields_ = [
        ("abi_version", I32),
        ("struct_size", I32),
        ("config_json", C.c_void_p),
        ("config_json_bytes", I64),
    ]


class Input(C.Structure):
    _fields_ = [
        ("abi_version", I32),
        ("struct_size", I32),
        ("layout", I32),
        ("scalar_type", I32),
        ("nstructures", I64),
        ("natoms", I64),
        ("positions", DP),
        ("positions_count", I64),
    ]


class Output(C.Structure):
    _fields_ = [
        ("abi_version", I32),
        ("struct_size", I32),
        ("energies", DP),
        ("energies_count", I64),
        ("forces", DP),
        ("forces_count", I64),
        ("structure_status", C.POINTER(I32)),
        ("structure_status_count", I64),
        ("structure_messages", C.c_void_p),
        ("structure_message_stride", I64),
        ("structure_messages_capacity", I64),
    ]


_PROCESS_LOCKS = {}
_LOCK = threading.Lock()


def _lock_for(path):
    with _LOCK:
        return _PROCESS_LOCKS.setdefault(str(path), threading.RLock())


def _ptr(a):
    return a.ctypes.data_as(DP)


def build_object(library, manifest, cache_directory=None):
    """Link .o plus declared wrappers. Never infer arbitrary Fortran signatures."""
    if not manifest:
        raise ValueError(
            ".o requires a PES manifest declaring its adapter and link dependencies"
        )
    mp = Path(manifest).resolve()
    config = (
        tomllib.loads(mp.read_text())
        if mp.suffix.lower() != ".json"
        else json.loads(mp.read_text())
    )
    base = mp.parent

    def resolve(p):
        p = Path(p)
        return p.resolve() if p.is_absolute() else (base / p).resolve()

    objects = [
        resolve(p) for p in config.get("objects", [str(Path(library).resolve())])
    ]
    wrappers = [resolve(p) for p in config.get("wrapper_sources", [])]
    paths = objects + wrappers
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError("PES object or wrapper source is missing")
    compiler = shlex.split(config.get("compiler", "cc"))
    if not compiler:
        raise ValueError("empty PES compiler")
    flags = config.get("compile_flags", ["-O3", "-fPIC"])
    libs = config.get("link_libraries", [])
    if any(not isinstance(s, str) for s in flags + libs):
        raise ValueError("link flags/libraries must be strings")
    digest = hashlib.sha256(mp.read_bytes())
    for p in paths:
        digest.update(str(p).encode())
        digest.update(p.read_bytes())
    try:
        ver = subprocess.run(
            compiler + ["--version"], check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PotentialError(f"PES compiler unavailable: {compiler[0]}") from exc
    digest.update(ver.encode())
    key = digest.hexdigest()
    cache = Path(
        cache_directory
        or os.environ.get(
            "PYRPMD_PES_CACHE", Path(tempfile.gettempdir()) / "pyrpmd-pes"
        )
    )
    target = cache / key / "libpes.so"
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    include = ["-I" + str(resolve(p)) for p in config.get("include_dirs", [])]
    library_dirs = ["-L" + str(resolve(p)) for p in config.get("library_dirs", [])]
    with tempfile.TemporaryDirectory(dir=target.parent, prefix="build-") as td:
        temporary = Path(td) / target.name
        command = (
            compiler
            + flags
            + ["-shared"]
            + include
            + [str(p) for p in paths]
            + library_dirs
            + ["-l" + p for p in libs]
            + ["-o", str(temporary)]
        )
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise PotentialError(
                "PES linking failed. Existing objects may need recompilation with -fPIC or missing runtime libraries.\n"
                + result.stderr[-6000:]
            )
        os.replace(temporary, target)
    return target


class CABIPotential(Potential):
    def __init__(self, library, config=None, system=None, manifest=None):
        path = Path(library).resolve()
        if path.suffix == ".o":
            path = build_object(path, manifest)
        if not path.is_file():
            raise ValueError(f"PES shared library does not exist: {path}")
        self.path = path
        self._context = C.c_void_p()
        self._lock = _lock_for(path)
        self.closed = False
        try:
            self.lib = C.CDLL(str(path))
        except OSError as exc:
            raise PotentialError(f"cannot load PES library {path}: {exc}") from exc
        prefix = "pyrpmd" if hasattr(self.lib, "pyrpmd_pes_abi_version_v1") else "nrpmd"

        def fn(name, args, result):
            try:
                f = getattr(self.lib, prefix + "_pes_" + name + "_v1")
            except AttributeError as exc:
                raise PotentialError(
                    "Unknown .so ABI: expected PyRPMD v1 symbols; use rpmdrate/python adapter for f2py"
                ) from exc
            f.argtypes = args
            f.restype = result
            return f

        version = fn("abi_version", [], I32)
        if version() != 1:
            raise PotentialError("unsupported PES ABI version")
        self._create = fn(
            "create", [C.POINTER(Options), C.POINTER(C.c_void_p), C.POINTER(Error)], I32
        )
        self._metadata = fn(
            "metadata",
            [C.c_void_p, C.c_void_p, I64, C.POINTER(I64), C.POINTER(Error)],
            I32,
        )
        self._evaluate = fn(
            "evaluate",
            [C.c_void_p, C.POINTER(Input), C.POINTER(Output), C.POINTER(Error)],
            I32,
        )
        self._destroy = fn("destroy", [C.c_void_p], None)
        payload = dict(config or {})

        def get(key, default=None):
            return (
                system.get(key, default)
                if isinstance(system, dict)
                else getattr(system, key, default)
            )

        if system is not None:
            symbols = list(get("symbols", []))
            payload.setdefault("natoms", len(symbols))
            payload.setdefault("symbols", symbols)
            numbers = get("atomic_numbers", None)
            if numbers is not None:
                payload.setdefault("atomic_numbers", np.asarray(numbers).tolist())
            for key in ("charge", "multiplicity"):
                if get(key) is not None:
                    payload.setdefault(key, get(key))
        raw = json.dumps(payload, sort_keys=True).encode()
        buf = C.create_string_buffer(raw)
        opts = Options(1, C.sizeof(Options), C.addressof(buf), len(raw))
        eb = C.create_string_buffer(4096)
        err = Error(C.addressof(eb), len(eb), 0)
        with self._lock:
            rc = self._create(C.byref(opts), C.byref(self._context), C.byref(err))
            if rc or not self._context:
                if self._context:
                    self._destroy(self._context)
                    self._context = C.c_void_p()
                raise PotentialError(
                    f"PES create failed ({rc}): {eb.value.decode(errors='replace')}"
                )
            try:
                needed = I64()
                rc = self._metadata(
                    self._context, None, 0, C.byref(needed), C.byref(err)
                )
                if rc not in (0, 4) or not 1 <= needed.value <= 16 * 1024 * 1024:
                    raise PotentialError("PES metadata sizing failed")
                mb = C.create_string_buffer(needed.value)
                rc = self._metadata(
                    self._context,
                    C.addressof(mb),
                    len(mb),
                    C.byref(needed),
                    C.byref(err),
                )
                if rc:
                    raise PotentialError(
                        "PES metadata failed: " + eb.value.decode(errors="replace")
                    )
                self.metadata = json.loads(mb.value.decode())
                for field, value in {
                    "abi_version": 1,
                    "layout": "B,N,3:C",
                    "dtype": "float64",
                    "position_unit": "bohr",
                    "energy_unit": "hartree",
                    "force_unit": "hartree/bohr",
                    "boundary_conditions": "nonperiodic",
                }.items():
                    if self.metadata.get(field) != value:
                        raise PotentialError(f"PES metadata {field} must be {value!r}")
                if self.metadata.get("schema") not in (
                    "pyrpmd.pes/1",
                    "next-rpmd.pes/1",
                ):
                    raise PotentialError("PES metadata schema mismatch")
                self.natoms = int(self.metadata["natoms"])
                if self.natoms < 1 or (
                    payload.get("natoms") and payload["natoms"] != self.natoms
                ):
                    raise PotentialError("PES atom count mismatch")
                if "atomic_numbers" in payload and payload[
                    "atomic_numbers"
                ] != self.metadata.get("atomic_numbers"):
                    raise PotentialError("PES element ordering mismatch")
                for key in ("charge", "multiplicity"):
                    if (
                        key in payload
                        and self.metadata.get("state", {}).get(key) != payload[key]
                    ):
                        raise PotentialError(f"PES {key} mismatch")
            except BaseException:
                self._destroy(self._context)
                self._context = C.c_void_p()
                raise

    def evaluate(self, x):
        if self.closed:
            raise PotentialError("PES context is closed")
        x = coordinates(x)
        if x.shape[1] != self.natoms:
            raise PotentialError("PES atom count mismatch")
        n = len(x)
        e = np.empty(n, dtype=np.float64)
        f = np.empty_like(x)
        s = np.empty(n, dtype=np.int32)
        messages = C.create_string_buffer(n * 512)
        eb = C.create_string_buffer(4096)
        err = Error(C.addressof(eb), len(eb), 0)
        inp = Input(1, C.sizeof(Input), 1, 1, n, self.natoms, _ptr(x), x.size)
        out = Output(
            1,
            C.sizeof(Output),
            _ptr(e),
            n,
            _ptr(f),
            f.size,
            s.ctypes.data_as(C.POINTER(I32)),
            n,
            C.addressof(messages),
            512,
            len(messages),
        )
        with self._lock:
            rc = self._evaluate(self._context, C.byref(inp), C.byref(out), C.byref(err))
        if rc:
            raise PotentialError(
                f"PES batch failed ({rc}): {eb.value.decode(errors='replace')}"
            )
        text = tuple(
            messages.raw[i * 512 : (i + 1) * 512]
            .split(b"\0", 1)[0]
            .decode(errors="replace")
            for i in range(n)
        )
        return validate_result(PESResult(e, f, s, text), x.shape, require_success=False)

    def close(self):
        if not self.closed:
            with self._lock:
                if self._context:
                    self._destroy(self._context)
                    self._context = C.c_void_p()
            self.closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
