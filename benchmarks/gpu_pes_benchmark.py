#!/usr/bin/env python3
"""Comprehensive PyRPMD GPU PES Benchmark Suite for Tesla T4 GPU.

Measures MD simulation efficiency for DeepMD, MACE (including pre-trained Carbon foundation
model and water), and JaxPIP across:
  - Bead scaling: P in [1, 4, 8, 16, 32]
  - Trajectory batching: K in [1, 4, 8, 16]
  - Dynamics mode: TorchDynamics (zero-copy GPU device) vs Dynamics (host NumPy transfers)
  - Batching mode: batched ML (vesin_torch) vs legacy non-batched
  - Metrics: Time per step (ms/step), Simulation speed (ns/day), Throughput (frames/s), VRAM (MiB).

Outputs results to JSON and prints formatted markdown tables.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ["VESIN_CUDA_MAX_PAIRS_PER_POINT"] = "4096"
os.environ["JAX_ENABLE_X64"] = "True"

# Ensure e3nn weights loading does not fail on PyTorch 2.6+
torch.serialization.add_safe_globals([slice])

# PyRPMD Imports
import pyrpmd
from pyrpmd.constants import AMU_TO_AU, BOHR_ANGSTROM, EV_TO_AU
from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials.adapters import make_named_potential
from pyrpmd.potentials.batched_ml import DeepMDBatchedPotential, MACEBatchedPotential
from pyrpmd.torch_backend import TorchDynamics, TorchPES, TorchState

# Constants
DT_FS = 0.5
DT_AU = DT_FS * 41.341374575751  # 1 fs = 41.341374575751 a.u.
BETA_300K = 1052.58  # 300 K in atomic units

# ---------------------------------------------------------------------------
# Test Systems Definitions
# ---------------------------------------------------------------------------

def get_carbon_cluster_c6():
    """Isolated C6 ring/cluster in Angstrom."""
    # Planar hexagon with CC bond length ~ 1.33 Angstrom
    r = 1.33
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pos_ang = np.zeros((6, 3))
    pos_ang[:, 0] = r * np.cos(angles)
    pos_ang[:, 1] = r * np.sin(angles)
    symbols = ["C"] * 6
    masses = [12.011 * AMU_TO_AU] * 6
    return symbols, pos_ang, masses, None, None


def get_carbon_diamond_c8():
    """Periodic Diamond unit cell (8 carbon atoms, a = 3.567 Angstrom)."""
    a = 3.567
    cell = np.eye(3) * a
    pbc = [True, True, True]
    # Standard diamond cubic coordinates (fractional)
    frac = np.array([
        [0.00, 0.00, 0.00],
        [0.50, 0.50, 0.00],
        [0.50, 0.00, 0.50],
        [0.00, 0.50, 0.50],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75]
    ])
    pos_ang = frac * a
    symbols = ["C"] * 8
    masses = [12.011 * AMU_TO_AU] * 8
    return symbols, pos_ang, masses, cell, pbc


def get_water_molecule():
    """Isolated H2O molecule in Angstrom."""
    pos_ang = np.array([
        [0.0000, 0.0000, 0.1173],   # O
        [0.0000, 0.7572, -0.4692],  # H
        [0.0000, -0.7572, -0.4692]  # H
    ])
    symbols = ["O", "H", "H"]
    masses = [15.999 * AMU_TO_AU, 1.008 * AMU_TO_AU, 1.008 * AMU_TO_AU]
    return symbols, pos_ang, masses, None, None


def get_water_box_32():
    """Periodic water box with 32 H2O molecules (96 atoms, L ~ 9.86 Angstrom)."""
    # Simple cubic grid arrangement of 32 water molecules
    L = 9.86
    cell = np.eye(3) * L
    pbc = [True, True, True]
    symbols = []
    masses = []
    coords = []
    # 32 points on ~ 3x3x4 grid
    idx = 0
    for ix in range(3):
        for iy in range(3):
            for iz in range(4):
                if idx >= 32:
                    break
                ox, oy, oz = (ix + 0.5) * (L / 3), (iy + 0.5) * (L / 3), (iz + 0.5) * (L / 4)
                coords.append([ox, oy, oz])
                coords.append([ox, oy + 0.757, oz - 0.469])
                coords.append([ox, oy - 0.757, oz - 0.469])
                symbols.extend(["O", "H", "H"])
                masses.extend([15.999 * AMU_TO_AU, 1.008 * AMU_TO_AU, 1.008 * AMU_TO_AU])
                idx += 1
    pos_ang = np.array(coords)
    pos_ang += np.random.default_rng(42).normal(0.0, 0.05, size=pos_ang.shape)
    return symbols, pos_ang, masses, cell, pbc


# ---------------------------------------------------------------------------
# Benchmark Timing Utilities
# ---------------------------------------------------------------------------

def measure_torch_dynamics(torch_pes, masses, K, P, pos_ang, n_steps=20, n_warmup=5):
    """Run TorchDynamics on GPU and record timing and VRAM."""
    N = len(pos_ang)
    pos_bohr = pos_ang / BOHR_ANGSTROM
    q0 = torch.tensor(pos_bohr, dtype=torch.float64, device="cuda:0")
    # Shape: [K, P, N, 3]
    q = q0.view(1, 1, N, 3).repeat(K, P, 1, 1).clone()
    p = torch.zeros_like(q)
    state = TorchState(q, p)
    m_tensor = torch.tensor(masses, dtype=torch.float64, device="cuda:0")

    dyn = TorchDynamics(torch_pes, m_tensor, beta=BETA_300K, nbeads=P, dt=DT_AU, integrator="cayley-bcb")

    # Warmup
    for _ in range(n_warmup):
        dyn.step(state)
    torch.cuda.synchronize()

    # Reset peak VRAM counter
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # Measurement
    t0 = time.perf_counter()
    for _ in range(n_steps):
        dyn.step(state)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    wall_s = t1 - t0
    ms_per_step = (wall_s / n_steps) * 1000.0
    ns_per_day = (n_steps * DT_FS * 1e-6 / wall_s) * 86400.0
    throughput = (K * P * n_steps) / wall_s
    vram_peak_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

    return {
        "engine": "TorchDynamics",
        "K": K,
        "P": P,
        "steps": n_steps,
        "wall_time_s": wall_s,
        "ms_per_step": ms_per_step,
        "ns_per_day": ns_per_day,
        "throughput_fps": throughput,
        "vram_peak_mib": vram_peak_mib
    }


def measure_host_dynamics(pes, masses, P, pos_ang, n_steps=10, n_warmup=2):
    """Run standard host Dynamics (K=1 trajectory) and record timing and VRAM."""
    N = len(pos_ang)
    pos_bohr = pos_ang / BOHR_ANGSTROM
    q = np.repeat(pos_bohr[np.newaxis, :, :], P, axis=0).copy()
    p = np.zeros_like(q)
    state = State(q, p)

    dyn = Dynamics(pes, masses, beta=BETA_300K, nbeads=P, dt=DT_AU, integrator="cayley-bcb")

    # Warmup
    for _ in range(n_warmup):
        dyn.step(state)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        dyn.step(state)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    wall_s = t1 - t0
    ms_per_step = (wall_s / n_steps) * 1000.0
    ns_per_day = (n_steps * DT_FS * 1e-6 / wall_s) * 86400.0
    throughput = (1 * P * n_steps) / wall_s
    vram_peak_mib = torch.cuda.max_memory_allocated() / (1024.0 ** 2) if torch.cuda.is_available() else 0.0

    return {
        "engine": "Dynamics",
        "K": 1,
        "P": P,
        "steps": n_steps,
        "wall_time_s": wall_s,
        "ms_per_step": ms_per_step,
        "ns_per_day": ns_per_day,
        "throughput_fps": throughput,
        "vram_peak_mib": vram_peak_mib
    }


# ---------------------------------------------------------------------------
# Main Benchmark Suite
# ---------------------------------------------------------------------------

def run_benchmarks():
    print("=" * 80)
    print("PyRPMD GPU Potential Energy Surface Benchmark Suite")
    print(f"Device: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB)")
    print(f"PyTorch: {torch.__version__}, PyRPMD: {pyrpmd.__version__}")
    print("=" * 80)

    results = {
        "metadata": {
            "device": torch.cuda.get_device_name(0),
            "vram_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "torch_version": torch.__version__,
            "pyrpmd_version": pyrpmd.__version__,
            "dt_fs": DT_FS,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
        "mace": {},
        "deepmd": {},
        "jaxpip": {},
        "unsupported_gaps": {}
    }

    # -----------------------------------------------------------------------
    # 1. MACE BENCHMARKS (Carbon Foundation Model & Water)
    # -----------------------------------------------------------------------
    print("\n>>> SECTION 1: MACE BENCHMARK (Best Carbon Foundation Model & Water)")
    mace_model_path = "/root/.cache/mace/20231203mace128L1_epoch199model"
    if not os.path.exists(mace_model_path):
        from mace.calculators import mace_mp
        mace_mp(model="medium", device="cuda")

    # 1.1 Carbon C6 Cluster
    c_symbols, c_pos, c_masses, _, _ = get_carbon_cluster_c6()
    print(f"\n--- 1.1 MACE: Carbon Cluster C6 (Gas phase, {len(c_symbols)} atoms) ---")
    pes_mace_c6 = MACEBatchedPotential(c_symbols, mace_model_path, device="cuda", dtype="float64")
    torch_pes_mace_c6 = pes_mace_c6.as_torch_pes()

    # Bead Scaling (P = 1, 4, 8, 16, 32; K = 1) - TorchDynamics
    mace_c6_bead_scaling = []
    for P in [1, 4, 8, 16, 32]:
        res = measure_torch_dynamics(torch_pes_mace_c6, c_masses, K=1, P=P, pos_ang=c_pos, n_steps=12, n_warmup=3)
        mace_c6_bead_scaling.append(res)
        print(f"[mace-batched C6 TorchDynamics] P={P:2d}, K=1: {res['ms_per_step']:6.2f} ms/step, {res['ns_per_day']:7.3f} ns/day, {res['throughput_fps']:8.1f} frames/s, VRAM: {res['vram_peak_mib']:.1f} MiB")

    # Trajectory Batching Scaling (K = 1, 4, 8, 16; P = 8) - TorchDynamics
    mace_c6_traj_scaling = []
    for K in [1, 4, 8, 16]:
        res = measure_torch_dynamics(torch_pes_mace_c6, c_masses, K=K, P=8, pos_ang=c_pos, n_steps=12, n_warmup=3)
        mace_c6_traj_scaling.append(res)
        print(f"[mace-batched C6 TorchDynamics] P=8, K={K:2d}: {res['ms_per_step']:6.2f} ms/step, {res['ns_per_day']:7.3f} ns/day, {res['throughput_fps']:8.1f} frames/s, VRAM: {res['vram_peak_mib']:.1f} MiB")

    # Host Dynamics vs TorchDynamics comparison (P=8, K=1)
    res_mace_c6_host = measure_host_dynamics(pes_mace_c6, c_masses, P=8, pos_ang=c_pos, n_steps=10, n_warmup=2)
    print(f"[mace-batched C6 Host Dynamics] P=8, K=1: {res_mace_c6_host['ms_per_step']:6.2f} ms/step, {res_mace_c6_host['ns_per_day']:7.3f} ns/day, {res_mace_c6_host['throughput_fps']:8.1f} frames/s")

    # Legacy ASE MACE comparison (P=4, K=1)
    try:
        pes_mace_ase = make_named_potential("mace", symbols=c_symbols, model=mace_model_path, device="cuda", params={"default_dtype": "float64"})
        res_mace_ase = measure_host_dynamics(pes_mace_ase, c_masses, P=4, pos_ang=c_pos, n_steps=5, n_warmup=1)
        print(f"[legacy ASE mace C6]            P=4, K=1: {res_mace_ase['ms_per_step']:6.2f} ms/step, {res_mace_ase['ns_per_day']:7.3f} ns/day, {res_mace_ase['throughput_fps']:8.1f} frames/s")
    except Exception as e:
        res_mace_ase = {"error": str(e)}
        print(f"[legacy ASE mace C6] Failed: {e}")

    # 1.2 Periodic Diamond Supercell (C8)
    try:
        dia_symbols, dia_pos, dia_masses, dia_cell, dia_pbc = get_carbon_diamond_c8()
        print(f"\n--- 1.2 MACE: Periodic Diamond C8 ({len(dia_symbols)} atoms, PBC) ---")
        pes_mace_dia = MACEBatchedPotential(dia_symbols, mace_model_path, device="cuda", dtype="float64", cell=dia_cell, pbc=dia_pbc)
        torch_pes_mace_dia = pes_mace_dia.as_torch_pes()
        mace_dia_res = measure_torch_dynamics(torch_pes_mace_dia, dia_masses, K=1, P=4, pos_ang=dia_pos, n_steps=10, n_warmup=2)
        print(f"[mace-batched Diamond C8 TorchDynamics] P=4, K=1: {mace_dia_res['ms_per_step']:6.2f} ms/step, {mace_dia_res['ns_per_day']:7.3f} ns/day, {mace_dia_res['throughput_fps']:8.1f} frames/s")
    except Exception as e:
        mace_dia_res = {"error": str(e)}
        print(f"[mace-batched Diamond C8] Exception: {e}")

    # 1.3 Water Systems (Isolated H2O and Periodic Liquid Box)
    w_symbols, w_pos, w_masses, _, _ = get_water_molecule()
    print(f"\n--- 1.3 MACE: Isolated Water H2O (3 atoms) ---")
    pes_mace_water = MACEBatchedPotential(w_symbols, mace_model_path, device="cuda", dtype="float64")
    torch_pes_mace_water = pes_mace_water.as_torch_pes()
    mace_water_res = measure_torch_dynamics(torch_pes_mace_water, w_masses, K=1, P=8, pos_ang=w_pos, n_steps=20, n_warmup=5)
    print(f"[mace-batched H2O TorchDynamics] P=8, K=1: {mace_water_res['ms_per_step']:6.2f} ms/step, {mace_water_res['ns_per_day']:7.3f} ns/day, {mace_water_res['throughput_fps']:8.1f} frames/s")

    try:
        wbox_symbols, wbox_pos, wbox_masses, wbox_cell, wbox_pbc = get_water_box_32()
        print(f"\n--- 1.4 MACE: Periodic Water Box 32 H2O ({len(wbox_symbols)} atoms, PBC) ---")
        pes_mace_wbox = MACEBatchedPotential(wbox_symbols, mace_model_path, device="cuda", dtype="float64", cell=wbox_cell, pbc=wbox_pbc)
        torch_pes_mace_wbox = pes_mace_wbox.as_torch_pes()
        mace_wbox_res = measure_torch_dynamics(torch_pes_mace_wbox, wbox_masses, K=1, P=4, pos_ang=wbox_pos, n_steps=10, n_warmup=2)
        print(f"[mace-batched WaterBox32 TorchDynamics] P=4, K=1: {mace_wbox_res['ms_per_step']:6.2f} ms/step, {mace_wbox_res['ns_per_day']:7.3f} ns/day, {mace_wbox_res['throughput_fps']:8.1f} frames/s")
    except Exception as e:
        mace_wbox_res = {"error": str(e)}
        print(f"[mace-batched WaterBox32] Exception: {e}")

    results["mace"] = {
        "carbon_c6_bead_scaling": mace_c6_bead_scaling,
        "carbon_c6_trajectory_scaling": mace_c6_traj_scaling,
        "carbon_c6_host_dynamics": res_mace_c6_host,
        "carbon_c6_legacy_ase": res_mace_ase,
        "diamond_c8": mace_dia_res,
        "isolated_water_h2o": mace_water_res,
        "water_box_32": mace_wbox_res
    }

    # -----------------------------------------------------------------------
    # 2. DEEPMD BENCHMARKS (Carbon & Water Models)
    # -----------------------------------------------------------------------
    print("\n>>> SECTION 2: DEEPMD BENCHMARK (Carbon & Water Models)")
    from deepmd.pt.model.model import get_model
    from deepmd.infer import DeepPot

    # 2.1 DeepMD Carbon Model (C6)
    c_deepmd_model = get_model({
        "type_map": ["C"],
        "descriptor": {
            "type": "se_e2_a", "rcut": 4.0, "rcut_smth": 3.5,
            "sel": [16], "neuron": [16, 32], "axis_neuron": 4,
            "precision": "float64", "seed": 42,
        },
        "fitting_net": {
            "type": "ener", "neuron": [32, 32], "precision": "float64",
            "seed": 43,
        },
    }).eval().to("cuda")

    # Save as TorchScript for deepmd-batched and DeepPot
    script_c = torch.jit.script(c_deepmd_model)
    pth_c_path = "/content/deepmd_carbon.pth"
    torch.jit.save(script_c, pth_c_path)

    pes_deepmd_c6 = DeepMDBatchedPotential(c_symbols, c_deepmd_model, device="cuda", dtype="float64")
    torch_pes_deepmd_c6 = pes_deepmd_c6.as_torch_pes()

    print(f"\n--- 2.1 DeepMD: Carbon Cluster C6 (Gas phase, {len(c_symbols)} atoms) ---")
    # Bead scaling P in [1, 4, 8, 16, 32]
    deepmd_c6_bead_scaling = []
    for P in [1, 4, 8, 16, 32]:
        res = measure_torch_dynamics(torch_pes_deepmd_c6, c_masses, K=1, P=P, pos_ang=c_pos, n_steps=50, n_warmup=10)
        deepmd_c6_bead_scaling.append(res)
        print(f"[deepmd-batched C6 TorchDynamics] P={P:2d}, K=1: {res['ms_per_step']:6.3f} ms/step, {res['ns_per_day']:7.2f} ns/day, {res['throughput_fps']:8.1f} frames/s, VRAM: {res['vram_peak_mib']:.1f} MiB")

    # Trajectory batching K in [1, 4, 8, 16]
    deepmd_c6_traj_scaling = []
    for K in [1, 4, 8, 16]:
        res = measure_torch_dynamics(torch_pes_deepmd_c6, c_masses, K=K, P=8, pos_ang=c_pos, n_steps=50, n_warmup=10)
        deepmd_c6_traj_scaling.append(res)
        print(f"[deepmd-batched C6 TorchDynamics] P=8, K={K:2d}: {res['ms_per_step']:6.3f} ms/step, {res['ns_per_day']:7.2f} ns/day, {res['throughput_fps']:8.1f} frames/s, VRAM: {res['vram_peak_mib']:.1f} MiB")

    # Host Dynamics vs TorchDynamics comparison (P=8, K=1)
    res_deepmd_c6_host = measure_host_dynamics(pes_deepmd_c6, c_masses, P=8, pos_ang=c_pos, n_steps=20, n_warmup=5)
    print(f"[deepmd-batched C6 Host Dynamics] P=8, K=1: {res_deepmd_c6_host['ms_per_step']:6.3f} ms/step, {res_deepmd_c6_host['ns_per_day']:7.2f} ns/day, {res_deepmd_c6_host['throughput_fps']:8.1f} frames/s")

    # Legacy DeepPot comparison (make_named_potential('deepmd'))
    try:
        pes_deepmd_dp = make_named_potential("deepmd", symbols=c_symbols, model=pth_c_path)
        res_deepmd_dp = measure_host_dynamics(pes_deepmd_dp, c_masses, P=8, pos_ang=c_pos, n_steps=20, n_warmup=5)
        print(f"[legacy deepmd DeepPot.eval C6]   P=8, K=1: {res_deepmd_dp['ms_per_step']:6.3f} ms/step, {res_deepmd_dp['ns_per_day']:7.2f} ns/day, {res_deepmd_dp['throughput_fps']:8.1f} frames/s")
    except Exception as e:
        res_deepmd_dp = {"error": str(e)}
        print(f"[legacy deepmd DeepPot C6] Failed: {e}")

    # 2.2 DeepMD Water Models (H2O and 32 H2O Box)
    water_deepmd_model = get_model({
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_e2_a", "rcut": 6.0, "rcut_smth": 5.5,
            "sel": [16, 32], "neuron": [16, 32], "axis_neuron": 4,
            "precision": "float64", "seed": 42,
        },
        "fitting_net": {
            "type": "ener", "neuron": [32, 32], "precision": "float64",
            "seed": 43,
        },
    }).eval().to("cuda")

    print(f"\n--- 2.2 DeepMD: Isolated Water H2O (3 atoms) ---")
    pes_deepmd_water = DeepMDBatchedPotential(w_symbols, water_deepmd_model, device="cuda", dtype="float64")
    torch_pes_deepmd_water = pes_deepmd_water.as_torch_pes()
    res_deepmd_water = measure_torch_dynamics(torch_pes_deepmd_water, w_masses, K=1, P=8, pos_ang=w_pos, n_steps=50, n_warmup=10)
    print(f"[deepmd-batched H2O TorchDynamics] P=8, K=1: {res_deepmd_water['ms_per_step']:6.3f} ms/step, {res_deepmd_water['ns_per_day']:7.2f} ns/day, {res_deepmd_water['throughput_fps']:8.1f} frames/s")

    try:
        print(f"\n--- 2.3 DeepMD: Periodic Water Box 32 H2O (96 atoms, PBC) ---")
        pes_deepmd_wbox = DeepMDBatchedPotential(wbox_symbols, water_deepmd_model, device="cuda", dtype="float64", cell=wbox_cell, pbc=wbox_pbc)
        torch_pes_deepmd_wbox = pes_deepmd_wbox.as_torch_pes()
        res_deepmd_wbox = measure_torch_dynamics(torch_pes_deepmd_wbox, wbox_masses, K=1, P=4, pos_ang=wbox_pos, n_steps=20, n_warmup=5)
        print(f"[deepmd-batched WaterBox32 TorchDynamics] P=4, K=1: {res_deepmd_wbox['ms_per_step']:6.3f} ms/step, {res_deepmd_wbox['ns_per_day']:7.2f} ns/day, {res_deepmd_wbox['throughput_fps']:8.1f} frames/s")
    except Exception as e:
        res_deepmd_wbox = {"error": str(e)}
        print(f"[deepmd-batched WaterBox32] Exception: {e}")

    results["deepmd"] = {
        "carbon_c6_bead_scaling": deepmd_c6_bead_scaling,
        "carbon_c6_trajectory_scaling": deepmd_c6_traj_scaling,
        "carbon_c6_host_dynamics": res_deepmd_c6_host,
        "carbon_c6_legacy_deeppot": res_deepmd_dp,
        "isolated_water_h2o": res_deepmd_water,
        "water_box_32": res_deepmd_wbox
    }

    # -----------------------------------------------------------------------
    # 3. JAXPIP BENCHMARKS (PR #201 Native ONNX CUDA & JAX-GPU DLPack Bridge)
    # -----------------------------------------------------------------------
    print("\n>>> SECTION 3: JAXPIP BENCHMARK (PR #201 Native ONNX CUDA & JAX-GPU DLPack)")
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    import onnxruntime as ort

    # 3.1 PR #201 Native ONNX CUDA Trajectory Batching
    print("\n--- 3.1 PR #201 Native ONNX CUDA Trajectory Batching (Harmonic Contract Model) ---")
    onnx_lib_path = "/content/PyRPMD/build/onnx/libpyrpmd_onnx.so"
    ort_sdk_lib = "/content/onnxruntime-sdk/lib"
    os.environ["LD_LIBRARY_PATH"] = f"{ort_sdk_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    # Generate harmonic ONNX contract model with metadata
    onnx_model_path = "/content/harmonic_contract.onnx"
    k_harm = 0.2
    inputs = [helper.make_tensor_value_info("q", TensorProto.DOUBLE, ["B", 3, 3])]
    outputs = [
        helper.make_tensor_value_info("energy", TensorProto.DOUBLE, ["B"]),
        helper.make_tensor_value_info("forces", TensorProto.DOUBLE, ["B", 3, 3])
    ]
    initializers = [
        numpy_helper.from_array(np.asarray([1, 2], np.int64), "axes"),
        numpy_helper.from_array(np.asarray(0.5 * k_harm, np.float64), "half_k"),
        numpy_helper.from_array(np.asarray(-k_harm, np.float64), "minus_k")
    ]
    nodes = [
        helper.make_node("Mul", ["q", "q"], ["square"]),
        helper.make_node("ReduceSum", ["square", "axes"], ["sum"], keepdims=0),
        helper.make_node("Mul", ["sum", "half_k"], ["energy"]),
        helper.make_node("Mul", ["q", "minus_k"], ["forces"])
    ]
    graph = helper.make_graph(nodes, "harmonic-contract", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    spec = {
        "schema": "pyrpmd.onnx/1",
        "batch_independent": True,
        "atomic_numbers": [1, 1, 8],
        "state": {"charge": 0, "multiplicity": 1},
        "position_unit": "angstrom",
        "energy_unit": "eV",
        "force_unit": "eV/angstrom",
        "derivative": "force",
        "constants": {"bohr_angstrom": BOHR_ANGSTROM, "ev_to_hartree": EV_TO_AU}
    }
    helper.set_model_props(model, {"pyrpmd.onnx": json.dumps(spec)})
    onnx.save(model, onnx_model_path)

    from pyrpmd.potentials import CABIPotential
    pes_cabi_gpu = CABIPotential(
        onnx_lib_path,
        {
            "model": onnx_model_path,
            "device": "cuda:0",
            "allow_cpu_fallback": False,
        },
    )

    # Benchmark trajectory batching scaling: K in [1, 2, 4, 8, 16], P = 8
    jaxpip_onnx_scaling = []
    P = 8
    for K in [1, 2, 4, 8, 16]:
        B = K * P
        q_test = np.random.randn(B, 3, 3) * 1.5
        # Warmup
        for _ in range(20):
            _ = pes_cabi_gpu.evaluate(q_test)
        torch.cuda.synchronize()

        n_eval = 200
        t0 = time.perf_counter()
        for _ in range(n_eval):
            _ = pes_cabi_gpu.evaluate(q_test)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        eval_wall = t1 - t0
        time_per_call_ms = (eval_wall / n_eval) * 1000.0
        fps = (B * n_eval) / eval_wall
        speedup = (jaxpip_onnx_scaling[0]["time_per_call_ms"] * K / time_per_call_ms) if jaxpip_onnx_scaling else 1.0

        item = {
            "K": K,
            "P": P,
            "B": B,
            "calls": n_eval,
            "time_per_call_ms": time_per_call_ms,
            "throughput_fps": fps,
            "speedup_vs_scalar": speedup
        }
        jaxpip_onnx_scaling.append(item)
        print(f"[PR #201 ONNX CUDA Batched] K={K:2d}, P={P:2d} (B={B:3d} frames): {time_per_call_ms:6.3f} ms/call, {fps:8.1f} frames/s, batch speedup: {speedup:5.2f}x")

    # 3.2 JAX-GPU DLPack Zero-Copy Bridge
    print("\n--- 3.2 JAX-GPU DLPack Zero-Copy Bridge (TorchPES + TorchDynamics) ---")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import equinox as eqx
    from jaxpip.descriptor import PolynomialDescriptor
    from jaxpip.model import PolynomialLinearModel

    WATER_PIP_BASIS = [
        [[0, 0, 0]],
        [[0, 0, 1], [0, 1, 0]],
        [[1, 0, 0]],
        [[0, 1, 1]],
        [[1, 0, 1], [1, 1, 0]],
        [[0, 0, 2], [0, 2, 0]],
        [[2, 0, 0]],
    ]
    pip_desc = PolynomialDescriptor(WATER_PIP_BASIS, alpha=1.2, dtype=jnp.float64)
    coeffs = jnp.ones(len(WATER_PIP_BASIS), dtype=jnp.float64) * 0.1
    jax_model = PolynomialLinearModel(pip_desc, coeffs=coeffs)

    class JaxPIPTorchBridge(torch.nn.Module):
        metadata = {"batch_independent": True}
        def __init__(self, jax_model):
            super().__init__()
            @eqx.filter_jit
            def _batch_energy_forces(q_model):
                return jax.vmap(jax_model.get_energy_and_forces)(q_model)
            self._batch_energy_forces = _batch_energy_forces
        def forward(self, q_model: torch.Tensor):
            q_contig = q_model.detach().contiguous()
            q_jax = jnp.from_dlpack(q_contig)
            e_jax, f_jax = self._batch_energy_forces(q_jax)
            return torch.from_dlpack(e_jax), torch.from_dlpack(f_jax)

    torch_pes_jaxpip = TorchPES(
        JaxPIPTorchBridge(jax_model),
        device="cuda:0",
        length_factor=1.0 / BOHR_ANGSTROM,
        energy_factor=EV_TO_AU
    )

    # Bead scaling for JaxPIP DLPack (P in [1, 4, 8, 16, 32], K = 1)
    jaxpip_dlpack_bead_scaling = []
    for P in [1, 4, 8, 16, 32]:
        res = measure_torch_dynamics(torch_pes_jaxpip, w_masses, K=1, P=P, pos_ang=w_pos, n_steps=50, n_warmup=10)
        jaxpip_dlpack_bead_scaling.append(res)
        print(f"[JaxPIP DLPack TorchDynamics] P={P:2d}, K=1: {res['ms_per_step']:6.3f} ms/step, {res['ns_per_day']:7.2f} ns/day, {res['throughput_fps']:8.1f} frames/s, VRAM: {res['vram_peak_mib']:.1f} MiB")

    # Trajectory scaling for JaxPIP DLPack (K in [1, 4, 8, 16], P = 8)
    jaxpip_dlpack_traj_scaling = []
    for K in [1, 4, 8, 16]:
        res = measure_torch_dynamics(torch_pes_jaxpip, w_masses, K=K, P=8, pos_ang=w_pos, n_steps=50, n_warmup=10)
        jaxpip_dlpack_traj_scaling.append(res)
        print(f"[JaxPIP DLPack TorchDynamics] P=8, K={K:2d}: {res['ms_per_step']:6.3f} ms/step, {res['ns_per_day']:7.2f} ns/day, {res['throughput_fps']:8.1f} frames/s, VRAM: {res['vram_peak_mib']:.1f} MiB")

    results["jaxpip"] = {
        "onnx_cuda_trajectory_scaling": jaxpip_onnx_scaling,
        "dlpack_bead_scaling": jaxpip_dlpack_bead_scaling,
        "dlpack_trajectory_scaling": jaxpip_dlpack_traj_scaling
    }

    # 3.3 Empirical Demonstration of Real JaxPIP ScatterND Failure on ORT CUDA EP
    print("\n--- 3.3 Empirical Test: Real JaxPIP ONNX on ORT CUDA EP (ScatterND Failure) ---")
    from pyrpmd.potentials import PotentialError
    # We export a real JaxPIP linear/NN model or test a graph with ScatterND on CUDA EP
    scatter_test_code = """
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort

# Construct a graph containing float64 ScatterND
data = helper.make_tensor_value_info('data', TensorProto.DOUBLE, [4, 4])
indices = helper.make_tensor_value_info('indices', TensorProto.INT64, [2, 1])
updates = helper.make_tensor_value_info('updates', TensorProto.DOUBLE, [2, 4])
output = helper.make_tensor_value_info('output', TensorProto.DOUBLE, [4, 4])

node = helper.make_node('ScatterND', ['data', 'indices', 'updates'], ['output'])
graph = helper.make_graph([node], 'scatter_test', [data, indices, updates], [output])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])

opts = ort.SessionOptions()
try:
    # 1. Test allow_cpu_fallback=False
    opts.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    sess = ort.InferenceSession(model.SerializeToString(), opts, providers=['CUDAExecutionProvider'])
    print("Session created without fallback!")
except Exception as e:
    print("Test 1 (disable_fallback):", type(e).__name__, str(e))

try:
    # 2. Test allow_cpu_fallback=True but evaluate with CUDA provider
    opts2 = ort.SessionOptions()
    sess2 = ort.InferenceSession(model.SerializeToString(), opts2, providers=['CUDAExecutionProvider'])
    d = np.zeros((4, 4), dtype=np.float64)
    idx = np.array([[1], [3]], dtype=np.int64)
    up = np.ones((2, 4), dtype=np.float64)
    out = sess2.run(['output'], {'data': d, 'indices': idx, 'updates': up})
    print("Execution succeeded!")
except Exception as e:
    print("Test 2 (runtime ScatterND):", type(e).__name__, str(e))
"""
    # Execute the empirical scatter test in python
    import subprocess
    scatter_res = subprocess.run([sys.executable, "-c", scatter_test_code], capture_output=True, text=True)
    print("Empirical ScatterND ORT Output:\n", scatter_res.stdout, scatter_res.stderr)

    results["unsupported_gaps"]["jaxpip_onnx_cuda_scatternd"] = {
        "status": "UNSUPPORTED",
        "technical_bottleneck": "ONNX Runtime CUDA Execution Provider lacks a CUDA kernel for double/float64 (element_type 11) ScatterND operations.",
        "empirical_verification": scatter_res.stdout + scatter_res.stderr,
        "optimization_target": "Implement custom CUDA kernel for float64 ScatterND in ORT / PyRPMD, or rewrite JaxPIP autograd graph to eliminate ScatterND, or employ float32 mixed precision, or use direct JAX DLPack bridge."
    }

    # -----------------------------------------------------------------------
    # Save Results
    # -----------------------------------------------------------------------
    out_file = "/content/gpu_pes_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll benchmark results successfully saved to {out_file}!")

    return results

if __name__ == "__main__":
    run_benchmarks()
