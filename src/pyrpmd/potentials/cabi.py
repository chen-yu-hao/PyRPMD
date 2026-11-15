"""Versioned host-memory and CUDA device-pointer C ABI loader."""

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
from .base import (
    BackendUnavailableError,
    Potential,
    PESResult,
    PotentialError,
    coordinates,
    validate_result,
)
from ..geometry import validate_cell_pbc

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


class InputV2(C.Structure):
    _fields_ = Input._fields_ + [
        ("cells", DP),
        ("cells_count", I64),
        ("pbc", C.POINTER(I32)),
        ("pbc_count", I64),
    ]


class OutputV2(C.Structure):
    _fields_ = Output._fields_ + [
        ("virials", DP),
        ("virials_count", I64),
    ]


class CudaInputV1(C.Structure):
    _fields_ = [
        ("cuda_abi_version", I32),
        ("struct_size", I32),
        ("layout", I32),
        ("scalar_type", I32),
        ("device_id", I32),
        ("flags", I32),
        ("cuda_stream", C.c_void_p),
        ("nstructures", I64),
        ("natoms", I64),
        ("d_positions", C.c_void_p),
        ("positions_count", I64),
        ("d_cells", C.c_void_p),
        ("cells_count", I64),
        ("d_pbc", C.c_void_p),
        ("pbc_count", I64),
    ]


class CudaOutputV1(C.Structure):
    _fields_ = [
        ("cuda_abi_version", I32),
        ("struct_size", I32),
        ("d_energies", C.c_void_p),
        ("energies_count", I64),
        ("d_forces", C.c_void_p),
        ("forces_count", I64),
        ("d_virials", C.c_void_p),
        ("virials_count", I64),
        ("d_structure_status", C.c_void_p),
        ("structure_status_count", I64),
    ]


_UNSET = object()
_IS_WINDOWS = os.name == "nt"
_PROCESS_LOCKS = {}
_LOCK = threading.Lock()


def _lock_for(path):
    with _LOCK:
        return _PROCESS_LOCKS.setdefault(str(path), threading.RLock())


def _unload_cdll(lib):
    """Release the Windows module handle so closed temporary DLLs can be unlinked."""
    if lib is not None and _IS_WINDOWS:
        handle = getattr(lib, "_handle", None)
        if handle:
            import _ctypes

            _ctypes.FreeLibrary(handle)


def _ptr(a):
    return a.ctypes.data_as(DP)


def _parse_cuda_device(value):
    if not isinstance(value, str) or not value.startswith("cuda:"):
        return None
    suffix = value[5:]
    if not suffix or len(suffix) >= 10 or not suffix.isdigit():
        return None
    if suffix != "0" and suffix[0] == "0":
        return None
    index = int(suffix)
    return index if 0 <= index <= 2**31 - 1 else None


def _symbol_numbers(symbols):
    """Canonical element identities, including isotope aliases in dict inputs."""
    from ..system import _Z

    if isinstance(symbols, str):
        raise PotentialError("PES symbols must be a sequence of element symbols")
    try:
        return [1 if symbol in ("D", "T") else _Z[symbol] for symbol in symbols]
    except (KeyError, TypeError) as exc:
        raise PotentialError("PES system contains an unknown element symbol") from exc


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
    def __init__(self, library, config=None, system=None, manifest=None, *, abi_version=None):
        if abi_version is not None and (type(abi_version) is not int or abi_version not in (1, 2)):
            raise ValueError("PES abi_version must be 1 or 2, or omitted for automatic selection")
        cell = system.get("cell") if isinstance(system, dict) else getattr(system, "cell", None)
        pbc = system.get("pbc") if isinstance(system, dict) else getattr(system, "pbc", None)
        self.cell, self.pbc = validate_cell_pbc(cell, pbc)
        if self.cell is not None:
            self.cell.flags.writeable = False
        self.pbc.flags.writeable = False
        self._geometry_cache = {}
        self._geometry_generation = 0
        self._cuda_base_geometry = None
        self._cuda_geometry_cache = {}
        if abi_version == 1 and self.pbc.any():
            raise PotentialError("PES C ABI v1 is nonperiodic and cannot accept a periodic System; use ABI v2")
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
        try:
            self._init_loaded_library(config, system, abi_version)
        except BaseException:
            self.closed = True
            lib, self.lib = self.lib, None
            _unload_cdll(lib)
            raise

    def _init_loaded_library(self, config, system, abi_version):
        self.abi_version = abi_version or (2 if hasattr(self.lib, "pyrpmd_pes_abi_version_v2") else 1)
        version_number = self.abi_version
        prefix = "pyrpmd" if version_number == 2 or hasattr(self.lib, "pyrpmd_pes_abi_version_v1") else "nrpmd"
        if version_number == 1 and self.pbc.any():
            raise PotentialError("PES C ABI v1 is nonperiodic and cannot accept a periodic System; use ABI v2")

        def fn(name, args, result):
            try:
                f = getattr(self.lib, prefix + "_pes_" + name + f"_v{version_number}")
            except AttributeError as exc:
                raise PotentialError(
                    f"Unknown .so ABI: expected PyRPMD v{version_number} symbols; use rpmdrate/python adapter for f2py"
                ) from exc
            f.argtypes = args
            f.restype = result
            return f

        version = fn("abi_version", [], I32)
        if version() != version_number:
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
            [C.c_void_p, C.POINTER(InputV2 if version_number == 2 else Input),
             C.POINTER(OutputV2 if version_number == 2 else Output), C.POINTER(Error)],
            I32,
        )
        self._destroy = fn("destroy", [C.c_void_p], None)

        has_cuda_ver = hasattr(self.lib, "pyrpmd_pes_cuda_abi_version_v1")
        has_cuda_eval = hasattr(self.lib, "pyrpmd_pes_evaluate_cuda_v1")
        if has_cuda_ver != has_cuda_eval:
            raise PotentialError(
                "Malformed PES CUDA ABI: library must export both "
                "pyrpmd_pes_cuda_abi_version_v1 and pyrpmd_pes_evaluate_cuda_v1"
            )
        self.cuda_abi_version = None
        self._evaluate_cuda = None
        if has_cuda_ver:
            cuda_version = self.lib.pyrpmd_pes_cuda_abi_version_v1
            cuda_version.argtypes = []
            cuda_version.restype = I32
            if cuda_version() != 1:
                raise PotentialError("unsupported PES CUDA ABI version")
            cuda_evaluate = self.lib.pyrpmd_pes_evaluate_cuda_v1
            cuda_evaluate.argtypes = [
                C.c_void_p,
                C.POINTER(CudaInputV1),
                C.POINTER(CudaOutputV1),
                C.POINTER(Error),
            ]
            cuda_evaluate.restype = I32
            self.cuda_abi_version = 1
            self._evaluate_cuda = cuda_evaluate

        payload = dict(config or {})

        def get(key, default=None):
            return (
                system.get(key, default)
                if isinstance(system, dict)
                else getattr(system, key, default)
            )

        if system is not None:
            raw_symbols = get("symbols", [])
            symbol_numbers = _symbol_numbers(raw_symbols)
            symbols = list(raw_symbols)
            payload.setdefault("natoms", len(symbols))
            payload.setdefault("symbols", symbols)
            numbers = get("atomic_numbers", None)
            if symbols:
                if numbers is not None and np.asarray(numbers).tolist() != symbol_numbers:
                    raise PotentialError("PES System symbols and atomic_numbers disagree")
                numbers = symbol_numbers
                if (payload["natoms"] != len(symbols)
                        or _symbol_numbers(payload["symbols"]) != symbol_numbers):
                    raise PotentialError("PES config cannot override System atom identities")
            if numbers is not None:
                numbers = np.asarray(numbers).tolist()
                if "atomic_numbers" in payload and payload["atomic_numbers"] != numbers:
                    raise PotentialError("PES config cannot override System atom identities")
                payload.setdefault("atomic_numbers", numbers)
            for key in ("charge", "multiplicity"):
                if get(key) is not None:
                    payload.setdefault(key, get(key))
        raw = json.dumps(payload, sort_keys=True).encode()
        buf = C.create_string_buffer(raw)
        opts = Options(version_number, C.sizeof(Options), C.addressof(buf), len(raw))
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
                    "abi_version": version_number,
                    "layout": "B,N,3:C",
                    "dtype": "float64",
                    "position_unit": "bohr",
                    "energy_unit": "hartree",
                    "force_unit": "hartree/bohr",
                }.items():
                    if self.metadata.get(field) != value:
                        raise PotentialError(f"PES metadata {field} must be {value!r}")
                schemas = ("pyrpmd.pes/1", "next-rpmd.pes/1") if version_number == 1 else ("pyrpmd.pes/2",)
                if self.metadata.get("schema") not in schemas:
                    raise PotentialError("PES metadata schema mismatch")
                boundary = self.metadata.get("boundary_conditions")
                if boundary not in (("nonperiodic",) if version_number == 1 else ("nonperiodic", "periodic")):
                    raise PotentialError("PES metadata boundary_conditions is unsupported")
                if self.pbc.any() and boundary != "periodic":
                    raise PotentialError("PES metadata declares a nonperiodic provider")
                capabilities = self.metadata.get("capabilities", {})
                if not isinstance(capabilities, dict):
                    raise PotentialError("PES capabilities must be an object")
                for cap_flag in ("host_memory", "cuda_device_memory"):
                    if cap_flag in capabilities and type(capabilities[cap_flag]) is not bool:
                        raise PotentialError(f"PES capabilities.{cap_flag} must be boolean")
                top_dev = self.metadata.get("execution_device")
                cap_dev = capabilities.get("execution_device")
                if top_dev is not None and cap_dev is not None and top_dev != cap_dev:
                    raise PotentialError(
                        "PES metadata execution_device and capabilities.execution_device disagree"
                    )
                execution_device = (
                    top_dev if top_dev is not None else (cap_dev if cap_dev is not None else "cpu")
                )
                if not isinstance(execution_device, str) or not execution_device:
                    raise PotentialError("PES metadata execution_device must be a non-empty string")
                self.metadata["execution_device"] = execution_device
                self.device = execution_device
                self._cuda_device_id = None
                if capabilities.get("cuda_device_memory") is True:
                    if self._evaluate_cuda is None:
                        raise PotentialError(
                            "PES metadata declares capabilities.cuda_device_memory=true "
                            "without exporting pyrpmd_pes_evaluate_cuda_v1"
                        )
                    cuda_id = _parse_cuda_device(execution_device)
                    if cuda_id is None:
                        raise PotentialError(
                            "PES capabilities.cuda_device_memory=true requires "
                            "execution_device='cuda:<nonnegative index>'"
                        )
                    self._cuda_device_id = cuda_id
                self.has_virial = False
                if version_number == 2:
                    if type(capabilities.get("native_batch", False)) is not bool:
                        raise PotentialError("PES ABI v2 capabilities.native_batch must be boolean")
                    for field, value in {
                        "cell_layout": "B,3,3:C:rows",
                        "pbc_layout": "B,3:C",
                    }.items():
                        if self.metadata.get(field) != value:
                            raise PotentialError(f"PES metadata {field} must be {value!r}")
                    if type(capabilities.get("virial")) is not bool:
                        raise PotentialError("PES ABI v2 capabilities.virial must be boolean")
                    self.has_virial = capabilities["virial"]
                    if self.has_virial:
                        for field, value in {
                            "virial_unit": "hartree",
                            "virial_layout": "B,3,3:C",
                            "virial_convention": "minus_dE_dstrain",
                        }.items():
                            if self.metadata.get(field) != value:
                                raise PotentialError(f"PES metadata {field} must be {value!r}")
                    self.metadata["periodic"] = boundary == "periodic"
                    self.metadata["supports_pbc"] = boundary == "periodic"
                    self.metadata["native_batch"] = capabilities.get("native_batch", False)
                    for field, default in (("thread_safety", "context_serial"),
                                           ("requires_bead_affinity", False)):
                        self.metadata.setdefault(field, capabilities.get(field, default))
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
                        raise PotentialError(
                            f"PES {key} mismatch: system/config declares {payload[key]!r}, "
                            f"provider declares {self.metadata.get('state', {}).get(key)!r}")
                if system is not None:
                    # Validate against the actual System too: user config may
                    # not override its composition or electronic state.
                    self.bind_system(system)
            except BaseException:
                self._destroy(self._context)
                self._context = C.c_void_p()
                raise

    def bind_system(self, system):
        """Bind a canonical-AU cell without recreating the compiled context.

        V2 sends the bound cell and flags on every evaluation, so a barostat
        may explicitly rebind between steps. Composition/electronic state
        cannot be changed on an existing context.
        """
        def get(key, default=None):
            return system.get(key, default) if isinstance(system, dict) else getattr(system, key, default)

        cell, pbc = validate_cell_pbc(get("cell"), get("pbc"))
        if pbc.any() and (self.abi_version != 2 or self.metadata.get("boundary_conditions") != "periodic"):
            raise PotentialError("PES C ABI provider does not support periodic boundary conditions")
        symbols = get("symbols")
        if symbols is not None and len(symbols) != self.natoms:
            raise PotentialError("PES atom count mismatch")
        numbers = get("atomic_numbers")
        if symbols is not None:
            derived_numbers = _symbol_numbers(symbols)
            if numbers is not None and np.asarray(numbers).tolist() != derived_numbers:
                raise PotentialError("PES System symbols and atomic_numbers disagree")
            numbers = derived_numbers
        if numbers is not None and np.asarray(numbers).tolist() != self.metadata.get("atomic_numbers"):
            raise PotentialError("PES element ordering mismatch")
        for key in ("charge", "multiplicity"):
            if get(key) is not None and get(key) != self.metadata.get("state", {}).get(key):
                raise PotentialError(
                    f"PES {key} mismatch: system declares {get(key)!r}, "
                    f"provider declares {self.metadata.get('state', {}).get(key)!r}")
        with self._lock:
            if self.closed:
                raise PotentialError("PES context is closed")
            if cell is not None:
                cell.flags.writeable = False
            pbc.flags.writeable = False
            self.cell, self.pbc = cell, pbc
            self._geometry_cache.clear()
            self._geometry_generation = getattr(self, "_geometry_generation", 0) + 1
            self._cuda_base_geometry = None
            if hasattr(self, "_cuda_geometry_cache"):
                self._cuda_geometry_cache.clear()

    def _batch_geometry(self, n, cell, pbc):
        # The bound geometry was validated at construction/rebind. A normal
        # fixed-cell dynamics step must not repeat B singular-value solves.
        if cell is _UNSET and pbc is _UNSET:
            cached = self._geometry_cache.get(n)
            if cached is None:
                cached = (
                    None if self.cell is None else np.broadcast_to(self.cell, (n, 3, 3)).copy(),
                    np.ascontiguousarray(np.broadcast_to(self.pbc, (n, 3)), dtype=np.int32),
                )
                # Bound the cache when a caller uses many different batches.
                self._geometry_cache.clear()
                self._geometry_cache[n] = cached
            return cached
        if cell is _UNSET:
            cell = self.cell
            if pbc is _UNSET:
                pbc = self.pbc
        if pbc is _UNSET:
            pbc = None
        if cell is None:
            cells = None
        else:
            cells = np.asarray(cell)
            if cells.shape == (3, 3):
                cells = np.broadcast_to(cells, (n, 3, 3))
            if cells.shape != (n, 3, 3):
                raise ValueError("PES cells must have shape [3,3] or [B,3,3]")
        flags = np.asarray(pbc) if pbc is not None else np.full((n, 3), cell is not None, dtype=bool)
        if flags.shape == (3,):
            flags = np.broadcast_to(flags, (n, 3))
        if flags.shape != (n, 3) or flags.dtype.kind != "b":
            raise ValueError("PES pbc must contain booleans with shape [3] or [B,3]")
        for i in range(n):
            validate_cell_pbc(None if cells is None else cells[i], flags[i])
        if flags.any() and self.metadata.get("boundary_conditions") != "periodic":
            raise PotentialError("PES metadata declares a nonperiodic provider")
        return (None if cells is None else np.ascontiguousarray(cells, dtype=np.float64),
                np.ascontiguousarray(flags, dtype=np.int32))

    def _ensure_cuda_base_geometry(self, torch, torch_device):
        if getattr(self, "abi_version", 1) != 2:
            return
        gen = getattr(self, "_geometry_generation", 0)
        base = getattr(self, "_cuda_base_geometry", None)
        if base is not None and base[0] == gen:
            return
        cell_1 = (
            None
            if self.cell is None
            else torch.as_tensor(
                np.array(self.cell, dtype=np.float64, copy=True),
                dtype=torch.float64,
                device=torch_device,
            )
            .reshape(1, 3, 3)
            .contiguous()
        )
        pbc_1 = (
            torch.as_tensor(self.pbc.astype(np.int32), dtype=torch.int32, device=torch_device)
            .reshape(1, 3)
            .contiguous()
        )
        # Synchronize the one-time geometry upload at bind/rebind so any
        # caller CUDA stream can read cell_1/pbc_1 without per-step sync.
        torch.cuda.current_stream(torch_device).synchronize()
        self._cuda_base_geometry = (gen, cell_1, pbc_1)

    def _cuda_batch_geometry(self, torch, torch_device, n, stream_handle, cell=_UNSET, pbc=_UNSET):
        if cell is _UNSET and pbc is _UNSET:
            self._ensure_cuda_base_geometry(torch, torch_device)
            gen, cell_1, pbc_1 = self._cuda_base_geometry
            cache = self._cuda_geometry_cache
            key = (gen, n, stream_handle)
            cached = cache.get(key)
            if cached is None:
                cells_b = None if cell_1 is None else cell_1.expand(n, 3, 3).contiguous()
                pbc_b = pbc_1.expand(n, 3).contiguous()
                if len(cache) >= 8:
                    cache.clear()
                cache[key] = (cells_b, pbc_b)
                cached = cache[key]
            return cached
        cells_np, flags_np = self._batch_geometry(n, cell, pbc)
        cells_t = (
            None
            if cells_np is None
            else torch.as_tensor(cells_np, dtype=torch.float64, device=torch_device).contiguous()
        )
        flags_t = torch.as_tensor(flags_np, dtype=torch.int32, device=torch_device).contiguous()
        return cells_t, flags_t

    def _validate_cuda_bridge_ready(self):
        if self.closed:
            raise PotentialError("PES context is closed")
        if (type(self) is not CABIPotential or "evaluate" in self.__dict__
                or CABIPotential.evaluate is not _CABI_EVALUATE):
            raise PotentialError("overridden C ABI provider cannot export a raw CUDA tensor binding")
        metadata = getattr(self, "metadata", {}) or {}
        capabilities = metadata.get("capabilities", {})
        if not isinstance(capabilities, dict):
            raise PotentialError("PES capabilities must be an object")
        if getattr(self, "_evaluate_cuda", None) is None or capabilities.get("cuda_device_memory") is not True:
            raise PotentialError(
                "C ABI provider does not support CUDA device-pointer evaluation "
                "(requires pyrpmd_pes_evaluate_cuda_v1 and capabilities.cuda_device_memory=true); "
                "no host/NumPy fallback"
            )
        if (metadata.get("batch_independent") is not True
                or metadata.get("trajectory_batch_safe") is False
                or any(metadata.get(k) for k in ("stateful", "coupled_frames", "ring_polymer_coupled"))
                or capabilities.get("stateful") is not False):
            raise PotentialError(
                "C ABI CUDA tensor binding requires batch_independent=true and capabilities.stateful=false"
            )
        if metadata.get("dtype") != "float64":
            raise PotentialError("C ABI CUDA tensor binding requires dtype='float64'")
        device_id = getattr(self, "_cuda_device_id", None)
        expected_dev = f"cuda:{device_id}" if device_id is not None else None
        if device_id is None or getattr(self, "device", None) != expected_dev or metadata.get("execution_device") != expected_dev:
            raise PotentialError("C ABI CUDA tensor binding requires execution_device='cuda:<index>'")
        try:
            import torch
        except ImportError as exc:
            raise BackendUnavailableError(
                "C ABI CUDA tensor evaluation requires PyTorch; install pyrpmd[gpu]"
            ) from exc
        if not torch.cuda.is_available():
            raise BackendUnavailableError("CUDA was requested for C ABI tensor binding, but PyTorch CUDA is unavailable")
        try:
            torch_device = torch.empty(0, device=expected_dev).device
        except (RuntimeError, AssertionError) as exc:
            raise BackendUnavailableError(
                f"requested CUDA device {expected_dev!r} is unavailable; no device fallback"
            ) from exc
        return torch, torch_device

    def _evaluate_cuda_impl(self, positions, torch, torch_device, *, cell=_UNSET, pbc=_UNSET, return_status=False):
        from ..torch_backend import TensorPESResult

        if self.closed:
            raise PotentialError("PES context is closed")
        if ("evaluate" in self.__dict__ or CABIPotential.evaluate is not _CABI_EVALUATE):
            raise PotentialError("overridden C ABI provider cannot execute raw CUDA tensor evaluation")
        if (not isinstance(positions, torch.Tensor) or positions.ndim != 3
                or positions.shape[0] < 1 or positions.shape[2] != 3):
            raise PotentialError("positions must be a nonempty tensor [batch,atom,3]")
        if positions.shape[1] != self.natoms:
            raise PotentialError("PES atom count mismatch")
        if positions.dtype != torch.float64 or positions.device != torch_device:
            raise PotentialError(
                f"positions must have dtype torch.float64 on {torch_device}; no implicit conversion"
            )
        if torch.is_autocast_enabled("cuda"):
            raise PotentialError("autocast is unsupported: C ABI CUDA evaluation requires float64 precision")

        q = positions if positions.is_contiguous() else positions.contiguous()
        b = q.shape[0]
        count = b * self.natoms * 3
        version_number = getattr(self, "abi_version", 1)
        has_virial = version_number == 2 and getattr(self, "has_virial", False)

        energy = torch.empty((b,), dtype=torch.float64, device=torch_device)
        forces = torch.empty((b, self.natoms, 3), dtype=torch.float64, device=torch_device)
        virial = torch.empty((b, 3, 3), dtype=torch.float64, device=torch_device) if has_virial else None
        status = torch.full((b,), -1, dtype=torch.int32, device=torch_device) if return_status else None
        stream_handle = int(torch.cuda.current_stream(torch_device).cuda_stream)

        eb = C.create_string_buffer(4096)
        err = Error(C.addressof(eb), len(eb), 0)

        with self._lock:
            if self.closed:
                raise PotentialError("PES context is closed")
            cells_t = pbc_t = None
            if version_number == 2:
                cells_t, pbc_t = self._cuda_batch_geometry(
                    torch, torch_device, b, stream_handle, cell=cell, pbc=pbc
                )
                d_cells = 0 if cells_t is None else cells_t.data_ptr()
                cells_count = 0 if cells_t is None else b * 9
                d_pbc = pbc_t.data_ptr()
                pbc_count = b * 3
            else:
                if cell is not _UNSET or pbc is not _UNSET:
                    raise PotentialError("Per-call cells/PBC require PES C ABI v2")
                d_cells = cells_count = d_pbc = pbc_count = 0

            inp = CudaInputV1(
                1,
                C.sizeof(CudaInputV1),
                1,
                1,
                torch_device.index,
                0,
                stream_handle,
                b,
                self.natoms,
                q.data_ptr(),
                count,
                d_cells,
                cells_count,
                d_pbc,
                pbc_count,
            )
            out = CudaOutputV1(
                1,
                C.sizeof(CudaOutputV1),
                energy.data_ptr(),
                b,
                forces.data_ptr(),
                count,
                0 if virial is None else virial.data_ptr(),
                0 if virial is None else b * 9,
                0 if status is None else status.data_ptr(),
                0 if status is None else b,
            )
            rc = self._evaluate_cuda(self._context, C.byref(inp), C.byref(out), C.byref(err))

        if rc:
            raise PotentialError(
                f"PES CUDA batch failed ({rc}): {eb.value.decode(errors='replace')}"
            )
        result = TensorPESResult(energy, forces, virial)
        return (result, status) if return_status else result

    def evaluate_tensor(self, positions, *, cell=_UNSET, pbc=_UNSET, return_status=False):
        """Evaluate device-resident float64 CUDA coordinates on the current CUDA stream."""
        torch, torch_device = self._validate_cuda_bridge_ready()
        return self._evaluate_cuda_impl(
            positions, torch, torch_device, cell=cell, pbc=pbc, return_status=return_status
        )

    def as_torch_pes(self):
        """Borrow this CUDA C ABI provider as a zero-copy ``TorchPES`` adapter.

        Requires an open provider exporting ``pyrpmd_pes_evaluate_cuda_v1`` with
        ``capabilities.cuda_device_memory=true``, ``capabilities.stateful=false``,
        ``batch_independent=true``, ``dtype='float64'``, and
        ``execution_device='cuda:<index>'``. Host-only C ABI libraries fail fast
        with no NumPy or host-staging fallback.
        """
        torch, torch_device = self._validate_cuda_bridge_ready()
        from ..torch_backend import TorchPES

        with self._lock:
            if self.closed:
                raise PotentialError("PES context is closed")
            self._ensure_cuda_base_geometry(torch, torch_device)

        version_number = getattr(self, "abi_version", 1)
        periodic = version_number == 2 and self.metadata.get("boundary_conditions") == "periodic"
        adapter_metadata = {
            **self.metadata,
            "provider": self.metadata.get("provider_name", "cabi-cuda"),
            "batch_independent": True,
            "trajectory_batch_safe": True,
            "dtype": "float64",
            "energy_dtype": "float64",
            "execution_device": str(torch_device),
            "periodic": periodic,
            "supports_pbc": periodic,
        }
        if version_number == 2 and getattr(self, "has_virial", False):
            adapter_metadata["virial_dtype"] = "float64"

        class _CABITorchPES(TorchPES):
            def __init__(self, provider):
                self.provider = provider
                self.device = torch_device
                self.metadata = adapter_metadata

            def evaluate_tensor(self, positions):
                return self.provider._evaluate_cuda_impl(positions, torch, self.device)

        return _CABITorchPES(self)

    def native_binding(self):
        from ..propagation import NativePESBinding, NativeUnavailableError
        if getattr(self, "abi_version", 1) != 1:
            raise NativeUnavailableError("C ABI v2 uses cell-aware batch evaluation, not the v1 fused kernel")
        if (type(self) is not CABIPotential or "evaluate" in self.__dict__
                or CABIPotential.evaluate is not _CABI_EVALUATE):
            raise NativeUnavailableError("overridden C ABI provider uses Python semantics")
        if self.metadata.get("capabilities", {}).get("stateful") is not False:
            raise NativeUnavailableError("C ABI provider must explicitly declare stateful=false")
        return NativePESBinding(self, self._evaluate, self._lock, self.natoms,
                                context=self._context.value or 0)

    def evaluate(self, x, *, cell=_UNSET, pbc=_UNSET):
        if self.closed:
            raise PotentialError("PES context is closed")
        if getattr(self, "metadata", {}).get("capabilities", {}).get("host_memory", True) is False:
            raise PotentialError("PES C ABI provider disables host_memory evaluation; use as_torch_pes() on CUDA")
        x = coordinates(x)
        if x.shape[1] != self.natoms:
            raise PotentialError("PES atom count mismatch")
        n = len(x)
        # Sentinels detect providers which forget a nominally successful frame.
        e = np.full(n, np.nan, dtype=np.float64)
        f = np.full_like(x, np.nan)
        s = np.full(n, -1, dtype=np.int32)
        messages = C.create_string_buffer(n * 512)
        eb = C.create_string_buffer(4096)
        err = Error(C.addressof(eb), len(eb), 0)
        version_number = getattr(self, "abi_version", 1)
        input_type = InputV2 if version_number == 2 else Input
        output_type = OutputV2 if version_number == 2 else Output
        inp = input_type(version_number, C.sizeof(input_type), 1, 1, n, self.natoms, _ptr(x), x.size)
        out = output_type(
            version_number,
            C.sizeof(output_type),
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
            # close() may have run while coordinates/output buffers were built.
            # Test the lifetime under the same lock used for native evaluation.
            if self.closed:
                raise PotentialError("PES context is closed")
            virial = None
            if version_number == 2:
                cells, flags = self._batch_geometry(n, cell, pbc)
                inp.cells = _ptr(cells) if cells is not None else DP()
                inp.cells_count = 0 if cells is None else cells.size
                inp.pbc = flags.ctypes.data_as(C.POINTER(I32))
                inp.pbc_count = flags.size
                if self.has_virial:
                    virial = np.full((n, 3, 3), np.nan, dtype=np.float64)
                    out.virials = _ptr(virial)
                    out.virials_count = virial.size
            elif cell is not _UNSET or pbc is not _UNSET:
                raise PotentialError("Per-call cells/PBC require PES C ABI v2")
            rc = self._evaluate(self._context, C.byref(inp), C.byref(out), C.byref(err))
        if rc:
            raise PotentialError(
                f"PES batch failed ({rc}): {eb.value.decode(errors='replace')}"
            )
        raw_messages = messages.raw
        text = tuple(
            raw_messages[i * 512 : (i + 1) * 512]
            .split(b"\0", 1)[0]
            .decode(errors="replace")
            for i in range(n)
        )
        return validate_result(PESResult(e, f, s, text, virial), x.shape, require_success=False)

    def close(self):
        with self._lock:
            if not self.closed:
                if self._context:
                    self._destroy(self._context)
                    self._context = C.c_void_p()
                self._cuda_base_geometry = None
                if hasattr(self, "_cuda_geometry_cache"):
                    self._cuda_geometry_cache.clear()
                # A previously exported binding retains the old context address.
                # Publish closed before releasing the lock so its acquire()
                # cannot admit that address after the context was destroyed.
                self.closed = True
                lib = getattr(self, "lib", None)
                self.lib = None
                _unload_cdll(lib)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# Binding export skips evaluate(), so only its original formula is eligible.
_CABI_EVALUATE = CABIPotential.evaluate
