"""Independent references for RPMDrate normal-mode interoperability.

Reference source: GreenGroup/RPMDrate commit
8c1828d2302aacf09a15faf5c7fc50a18d4c113e, rpmdrate/_math.f90
(rfft/irfft) and _main.f90 (free_ring_polymer_step, verlet_step).
This is a new implementation of the mathematical convention, not vendored
RPMDrate code. Optional ctypes calls exercise the actual FFTW R2HC/HC2R ABI.
BCOCB is our extension; the inspected upstream uses exact harmonic rotations.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math

import numpy as np
from scipy.linalg import sqrtm

REFERENCE_COMMIT = "8c1828d2302aacf09a15faf5c7fc50a18d4c113e"


def real_transform(P):
    """Direct trigonometric forward transform; no FFT or production helpers."""
    j = np.arange(P)
    indices = list(range(P // 2 + 1)) + list(range((P - 1) // 2, 0, -1))
    rows = [np.cos(2 * np.pi * k * j / P) / math.sqrt(P)
            for k in range(P // 2 + 1)]
    rows += [-np.sin(2 * np.pi * k * j / P) / math.sqrt(P)
             for k in range((P - 1) // 2, 0, -1)]
    weights = np.full(P, 2.); weights[0] = 1.
    if P % 2 == 0:
        weights[P // 2] = 1.
    return np.array(rows), weights, np.array(indices)


def fftw_transform(values, *, inverse=False):
    """Real FFTW transform of the LAST axis with RPMDrate normalization.

    Used for validation only; copies/loops on the CPU intentionally. Production
    uses torch.fft and never calls this helper. FFTW plans are always released.
    """
    library = ctypes.util.find_library("fftw3")
    if not library:
        raise RuntimeError("optional FFTW3 shared library not installed")
    lib = ctypes.CDLL(library)
    pointer = ctypes.POINTER(ctypes.c_double)
    lib.fftw_plan_r2r_1d.argtypes = [ctypes.c_int, pointer, pointer, ctypes.c_int, ctypes.c_uint]
    lib.fftw_plan_r2r_1d.restype = ctypes.c_void_p
    lib.fftw_execute.argtypes = [ctypes.c_void_p]
    lib.fftw_execute.restype = None
    lib.fftw_destroy_plan.argtypes = [ctypes.c_void_p]
    lib.fftw_destroy_plan.restype = None
    values = np.asarray(values, dtype=np.float64)
    P = values.shape[-1]
    source, target = np.zeros(P), np.zeros(P)
    # FFTW_R2HC=0 / FFTW_HC2R=1, FFTW_ESTIMATE=64; matches the source routines.
    plan = lib.fftw_plan_r2r_1d(P, source.ctypes.data_as(pointer),
                               target.ctypes.data_as(pointer), int(inverse), 64)
    if not plan:
        raise RuntimeError("FFTW plan creation failed")
    output = np.empty(values.shape, dtype=np.float64)
    try:
        for i, row in enumerate(values.reshape(-1, P)):
            source[:] = row
            lib.fftw_execute(plan)
            output.reshape(-1, P)[i] = target / math.sqrt(P)
    finally:
        lib.fftw_destroy_plan(plan)
    return output


def exact_free(q, p, masses, beta, dt):
    """RPMDrate exact free-ring map, actual FFTW transforms, [T,P,N,3]."""
    P = q.shape[1]
    qk = np.moveaxis(fftw_transform(np.moveaxis(q, 1, -1)), -1, 1)
    pk = np.moveaxis(fftw_transform(np.moveaxis(p, 1, -1)), -1, 1)
    _, _, indices = real_transform(P)
    omega = 2 * P / beta * np.sin(np.pi * indices / P)
    a = np.cos(dt * omega)[None, :, None, None]
    b = (dt * np.sinc(dt * omega / np.pi))[None, :, None, None]
    c = (omega * np.sin(dt * omega))[None, :, None, None]
    mass = np.asarray(masses)[None, None, :, None]
    qnew, pnew = a * qk + b * pk / mass, a * pk - c * mass * qk
    inverse = lambda x: np.moveaxis(fftw_transform(np.moveaxis(x, 1, -1), inverse=True), -1, 1)
    return inverse(qnew), inverse(pnew)


def bcocb_reference(q, p, bead_noise, masses, beta, dt, stiffness, gamma_centroid=1.):
    """Independent dense trigonometric/matrix-square-root BCOCB step.

    stiffness[N,N] couples atoms in each Cartesian direction. Noise is a real
    bead-space unit Gaussian field; the transform supplies the packed metric.
    No code or coefficients from either PyRPMD backend are used.
    """
    P = q.shape[1]
    F, metric, index = real_transform(P)
    inverse = F.T * metric
    force = lambda x: -np.einsum("nm,tpmc->tpnc", stiffness, x)
    p = p + dt / 2 * force(q)
    qk = np.einsum("kp,tpnc->tknc", F, q)
    pk = np.einsum("kp,tpnc->tknc", F, p)
    nk = np.einsum("kp,tpnc->tknc", F, bead_noise)
    for k, w in enumerate(2 * P / beta * np.sin(np.pi * index / P)):
        gamma = gamma_centroid if k == 0 else 2 * w
        decay, amplitude = np.exp(-dt * gamma), np.sqrt(-np.expm1(-2 * dt * gamma))
        for atom, mass in enumerate(masses):
            A = np.array([[0., 1 / mass], [-mass * w * w, 0.]])
            I = np.eye(2)
            root = np.real_if_close(sqrtm(np.linalg.solve(I - dt * A / 2, I + dt * A / 2)))
            phase = np.stack((qk[:, k, atom], pk[:, k, atom]), axis=-1) @ root.T
            phase[..., 1] = decay * phase[..., 1] + amplitude * np.sqrt(mass * P / beta) * nk[:, k, atom]
            phase = phase @ root.T
            qk[:, k, atom], pk[:, k, atom] = phase[..., 0], phase[..., 1]
    q = np.einsum("pk,tknc->tpnc", inverse, qk)
    p = np.einsum("pk,tknc->tpnc", inverse, pk)
    return q, p + dt / 2 * force(q)
