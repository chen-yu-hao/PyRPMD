"""Batched exact fixed-nuclear-coordinate mapping flow, in atomic units.

All backends integrate both the electronic vector and its nuclear impulse.
``native`` uses batched LAPACK diagonalization followed by a C++/OpenMP
contraction kernel. ``torch`` uses device-resident tensor operations; the
NumPy-facing wrapper explicitly transfers its results back to the host.
"""
from __future__ import annotations

import importlib.util
import numpy as np


def _scalar(value, name):
    a = np.asarray(value)
    if a.ndim or a.dtype.kind not in "fiu" or not np.isfinite(a):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(a)


def _threads(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("threads must be a positive integer")
    return int(value)


def validate_electronic_backend(backend="numpy", device="cpu"):
    """Validate a requested backend without silently choosing another one."""
    if backend not in ("numpy", "native", "torch"):
        raise ValueError("electronic_backend must be numpy, native or torch")
    if not isinstance(device, str) or not device:
        raise ValueError("electronic_device must be a nonempty device string")
    if backend != "torch":
        if device != "cpu":
            raise ValueError("only the torch electronic backend accepts a non-CPU device")
        if backend == "native":
            try:
                from . import _electronic
            except ImportError as error:
                raise RuntimeError("native electronic extension is unavailable; rebuild PyRPMD") from error
            if getattr(_electronic, "MAPPING_FLOW_API", None) != 1:
                raise RuntimeError("native electronic extension API mismatch; rebuild PyRPMD")
        return
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("torch electronic backend requires the optional PyTorch dependency") from error
    try:
        selected = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise ValueError("invalid electronic_device") from error
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("electronic_device supports cpu or cuda[:index] with complex128")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        if selected.index is not None and selected.index >= torch.cuda.device_count():
            raise ValueError("CUDA device index is out of range")


def electronic_backend_capabilities():
    """Report installation and hardware availability; no performance claim."""
    native = False
    try:
        from . import _electronic
        native = getattr(_electronic, "MAPPING_FLOW_API", None) == 1
    except ImportError:
        pass
    torch_installed = importlib.util.find_spec("torch") is not None
    cuda = False
    if torch_installed:
        import torch
        cuda = torch.cuda.is_available()
    return {"numpy": True, "native": native, "torch": torch_installed,
            "cuda_available": cuda, "precision": "float64/complex128"}


def _inputs(z, matrix, gradients):
    z, v, g = (np.asarray(a) for a in (z, matrix, gradients))
    if any(a.dtype.kind not in "fciu" or not np.isfinite(a).all() for a in (z, v, g)):
        raise ValueError("electronic inputs must be finite numerical arrays")
    if z.ndim < 1 or min(z.shape) < 1:
        raise ValueError("z must have nonempty shape [..., S]")
    leading, states = z.shape[:-1], z.shape[-1]
    if v.shape != (*leading, states, states):
        raise ValueError("matrix must have shape [..., S, S] matching z")
    if (g.ndim != len(leading)+4 or g.shape[:len(leading)+2] != v.shape
            or g.shape[-2] < 1 or g.shape[-1] != 3):
        raise ValueError("gradients must have shape [..., S, S, N, 3]")
    axis = len(leading)
    if not np.allclose(v, v.swapaxes(-1, -2).conj(), rtol=0., atol=1e-12):
        raise ValueError("matrix must be Hermitian")
    if not np.allclose(g, g.swapaxes(axis, axis+1).conj(), rtol=0., atol=1e-12):
        raise ValueError("matrix gradients must be Hermitian")
    v = 0.5*(v+v.swapaxes(-1, -2).conj())
    g = 0.5*(g+g.swapaxes(axis, axis+1).conj())
    return (np.ascontiguousarray(z, dtype=np.complex128),
            np.ascontiguousarray(v, dtype=np.complex128),
            np.ascontiguousarray(g, dtype=np.complex128), leading, states)


def _numpy_flow(z, matrix, gradients, duration, gamma):
    energies, vectors = np.linalg.eigh(matrix)
    ze = np.einsum("bsi,bs->bi", vectors.conj(), z)
    phase = 0.5*duration*(energies[:, :, None]-energies[:, None, :])
    integral = duration*np.sinc(phase/np.pi)*np.exp(1j*phase)
    density_e = ze.conj()[:, :, None]*ze[:, None, :]*integral
    density = (vectors.conj() @ density_e) @ vectors.swapaxes(1, 2)
    delta = -0.5*np.einsum("bij,bijac->bac", density, gradients).real
    delta += 0.5*gamma*duration*np.einsum("biiac->bac", gradients).real
    rotated = np.einsum("bsi,bi->bs", vectors, ze*np.exp(-1j*duration*energies))
    return rotated, delta


def electronic_mapping_flow(z, matrix, gradients, duration, *, gamma=1.0,
                            backend="numpy", device="cpu", threads=1):
    """Return ``(z_new, delta_p)`` for the exact mapping Hamiltonian flow.

    The Hamiltonian is ``(z^dagger V z - gamma*Tr(V))/2``. Add ``delta_p``
    to the nuclear momentum. Shapes are ``z[...,S]``, ``V[...,S,S]`` and
    positive gradients ``G[...,S,S,N,3]``; leading dimensions may include
    trajectory and bead. Degenerate eigenvalues use the analytic sinc limit.
    Inputs are not mutated. All outputs retain float64/complex128 precision.
    """
    duration, gamma = _scalar(duration, "duration"), _scalar(gamma, "gamma")
    threads = _threads(threads)
    validate_electronic_backend(backend, device)
    z, v, g, leading, states = _inputs(z, matrix, gradients)
    atoms = g.shape[-2]
    flat_z, flat_v = z.reshape(-1, states), v.reshape(-1, states, states)
    flat_g = g.reshape(-1, states, states, atoms, 3)
    if backend == "numpy":
        rotated, delta = _numpy_flow(flat_z, flat_v, flat_g, duration, gamma)
    elif backend == "native":
        from . import _electronic
        energies, vectors = np.linalg.eigh(flat_v)
        rotated = np.empty_like(flat_z)
        delta = np.empty((len(flat_z), atoms, 3), dtype=np.float64)
        _electronic.mapping_flow(
            np.ascontiguousarray(energies), np.ascontiguousarray(vectors), flat_z,
            flat_g.reshape(len(flat_z), states, states, 3*atoms), duration, gamma,
            rotated, delta.reshape(len(flat_z), 3*atoms), threads)
    else:
        import torch
        tensors = [torch.as_tensor(a, dtype=torch.complex128, device=device)
                   for a in (flat_z, flat_v, flat_g)]
        tz, tp = torch_electronic_mapping_flow(*tensors, duration, gamma=gamma)
        rotated, delta = tz.cpu().numpy(), tp.cpu().numpy()
    if not np.isfinite(rotated).all() or not np.isfinite(delta).all():
        raise FloatingPointError("electronic mapping flow produced nonfinite output")
    return rotated.reshape(*leading, states), delta.reshape(*leading, atoms, 3)


def torch_electronic_mapping_flow(z, matrix, gradients, duration, *, gamma=1.0):
    """Tensor-in/tensor-out flow that keeps its arrays on one CPU/CUDA device.

    Requires complex128 tensors on the same device. This deterministic
    dynamics primitive disables autograd; it does not differentiate through
    an eigensystem. Validation reductions may synchronize a CUDA device.
    """
    import torch
    duration, gamma = _scalar(duration, "duration"), _scalar(gamma, "gamma")
    arrays = (z, matrix, gradients)
    if any(not isinstance(a, torch.Tensor) for a in arrays):
        raise TypeError("torch flow requires tensors")
    if any(a.dtype != torch.complex128 or a.device != z.device for a in arrays):
        raise ValueError("torch flow requires complex128 tensors on one device")
    validate_electronic_backend("torch", str(z.device))
    if z.ndim < 1 or min(z.shape) < 1:
        raise ValueError("z must have nonempty shape [..., S]")
    leading, states = tuple(z.shape[:-1]), z.shape[-1]
    if tuple(matrix.shape) != (*leading, states, states):
        raise ValueError("matrix shape mismatch")
    if (gradients.ndim != len(leading)+4
            or tuple(gradients.shape[:-2]) != tuple(matrix.shape)
            or gradients.shape[-2] < 1 or gradients.shape[-1] != 3):
        raise ValueError("gradient shape mismatch")
    if any(not bool(torch.isfinite(a).all()) for a in arrays):
        raise ValueError("electronic inputs must be finite")
    axis = len(leading)
    if not torch.allclose(matrix, matrix.mH, rtol=0., atol=1e-12):
        raise ValueError("matrix must be Hermitian")
    if not torch.allclose(gradients, gradients.transpose(axis, axis+1).conj(), rtol=0., atol=1e-12):
        raise ValueError("matrix gradients must be Hermitian")
    with torch.no_grad():
        v = 0.5*(matrix+matrix.mH)
        g = 0.5*(gradients+gradients.transpose(axis, axis+1).conj())
        energies, vectors = torch.linalg.eigh(v)
        ze = torch.einsum("...si,...s->...i", vectors.conj(), z)
        phase = 0.5*duration*(energies[..., :, None]-energies[..., None, :])
        integral = duration*torch.sinc(phase/np.pi)*torch.exp(1j*phase)
        density_e = ze.conj()[..., :, None]*ze[..., None, :]*integral
        density = (vectors.conj() @ density_e) @ vectors.transpose(-1, -2)
        delta = -0.5*torch.einsum("...ij,...ijac->...ac", density, g).real
        delta += 0.5*gamma*duration*torch.einsum("...iiac->...ac", g).real
        rotated = torch.einsum("...si,...i->...s", vectors, ze*torch.exp(-1j*duration*energies))
        if not bool(torch.isfinite(rotated).all()) or not bool(torch.isfinite(delta).all()):
            raise FloatingPointError("electronic mapping flow produced nonfinite output")
        return rotated, delta
