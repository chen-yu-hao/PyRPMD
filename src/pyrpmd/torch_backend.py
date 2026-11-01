"""Opt-in, device-resident real-mass ring-polymer dynamics (atomic units).

This Python API is deliberately separate from the NumPy workflow/checkpoint
contract. State has shape [trajectories, beads, atoms, 3]. Every physical PES
component receives all its independent configurations in one tensor call.

No ASE calculator is silently treated as a batched model. Regional contraction
requires an explicit additive energy decomposition; it is not force masking.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
from typing import Sequence

import torch
from torch import Tensor


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _independent(metadata, declared=None):
    if not isinstance(metadata, dict):
        raise ValueError("model metadata must be a mapping")
    capabilities = metadata.get("capabilities", {}) or {}
    if not isinstance(capabilities, dict):
        raise ValueError("model capabilities must be a mapping")
    promise = metadata.get("batch_independent") if declared is None else declared
    if (promise is not True or metadata.get("batch_independent") is False
            or metadata.get("stateful") or metadata.get("coupled_frames")
            or metadata.get("ring_polymer_coupled") or capabilities.get("stateful")):
        raise ValueError("a deterministic, batch-independent model must be explicitly declared")


def _tensor(value, name, shape, device):
    if (not isinstance(value, Tensor) or value.dtype != torch.float64
            or value.device != device or tuple(value.shape) != tuple(shape)):
        raise ValueError(f"{name} must be a float64 tensor {tuple(shape)} on {device}")


@dataclass(frozen=True)
class TensorPESResult:
    """Energy and negative energy gradients, in hartree and hartree/bohr."""
    energy: Tensor
    forces: Tensor
    virial: Tensor | None = None


class TorchPES:
    """Batched torch.nn.Module adapter with no implicit host transfer.

    ``forward(q)`` (or ``forward(q, cell, pbc)`` for a periodic model) must
    return independent-frame energies [F] / [F,1], (energies, forces), or a
    mapping with energy/energies and optional forces, virial, or stress.
    Forces are negative derivatives in the MODEL's units. ``length_factor``
    is bohr per model length unit and ``energy_factor`` hartree per model
    energy unit. A model must explicitly promise batch independence, either
    here or through metadata['batch_independent'] = True.

    Construction moves the module and sets eval mode. Do not share it with
    a concurrently training model. Parameters' requires_grad flags are not
    changed; autograd.grad does not populate their .grad fields.
    """
    def __init__(self, model, *, device="cpu", batch_independent=None,
                 length_factor=1.0, energy_factor=1.0, cell=None, pbc=None,
                 periodic=False, virial="none"):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module; ASE calculators need an explicit model adapter")
        metadata = getattr(model, "metadata", {}) or {}
        _independent(metadata, batch_independent)
        if not isinstance(periodic, bool):
            raise ValueError("periodic must be a boolean")
        if virial not in ("none", "provided", "autograd", "auto"):
            raise ValueError("virial must be none, provided, autograd, or auto")
        requested = torch.device(device)
        if requested.type not in ("cpu", "cuda"):
            raise ValueError("this float64 backend supports CPU and CUDA devices")
        # Resolve cuda -> cuda:N once, before entering the hot loop.
        self.device = torch.empty(0, device=requested).device
        self.length_factor = _positive(length_factor, "length_factor")
        self.energy_factor = _positive(energy_factor, "energy_factor")
        self.periodic, self.virial_mode = periodic, virial
        flags = torch.as_tensor([False] * 3 if pbc is None else pbc, device=self.device)
        if flags.dtype != torch.bool or flags.shape != (3,):
            raise ValueError("pbc must contain three booleans")
        self.pbc = flags.detach().clone()
        self.cell = None
        if cell is not None:
            self.cell = torch.as_tensor(cell, dtype=torch.float64, device=self.device).detach().clone()
            if (self.cell.shape != (3, 3) or not torch.isfinite(self.cell).all().item()
                    or torch.linalg.det(self.cell).abs().item() == 0):
                raise ValueError("cell must be a finite nonsingular [3,3] matrix in bohr")
            self.cell /= self.length_factor
        if periodic and self.cell is None:
            raise ValueError("periodic=True requires a cell")
        if self.pbc.any().item() and not periodic:
            raise ValueError("periodic boundary flags require periodic=True")
        self._volume = None if self.cell is None else torch.linalg.det(self.cell).abs()
        self.model = model.to(device=self.device, dtype=torch.float64).eval()
        self.metadata = {"provider": "torch-tensor", "batch_independent": True,
                         "trajectory_batch_safe": True, "execution_device": str(self.device)}

    @classmethod
    def from_provider(cls, provider, *, batch_independent=None):
        """Adapt an existing optional.TorchPotential once, not once per step.

        NumPy factory delegates cannot satisfy this contract and are rejected.
        The module is shared; keep the original provider alive and do not
        mutate its model/cell while using this fixed-cell adapter.
        """
        metadata = getattr(provider, "metadata", {}) or {}
        if (not isinstance(metadata, dict) or metadata.get("provider") != "torch"
                or getattr(provider, "closed", False)
                or getattr(provider, "delegate", None) is not None):
            raise ValueError("requires an open, module-backed TorchPotential, not a NumPy delegate")
        _independent(metadata, batch_independent)
        if batch_independent is None:
            batch_independent = metadata.get("batch_independent")
        return cls(provider.model, device=provider.device,
                   batch_independent=batch_independent,
                   length_factor=provider.length_factor, energy_factor=provider.energy_factor,
                   cell=provider.cell, pbc=provider.pbc, periodic=provider.periodic,
                   virial=provider.virial_mode)

    def evaluate_tensor(self, positions):
        if not isinstance(positions, Tensor) or positions.ndim != 3 or min(positions.shape) < 1:
            raise ValueError("positions must be a nonempty tensor [frames,atoms,3]")
        frames, atoms, _ = positions.shape
        _tensor(positions, "positions", (frames, atoms, 3), self.device)
        if torch.is_inference_mode_enabled():
            raise ValueError("use no_grad, not inference_mode: energy-only models need coordinate gradients")
        with torch.enable_grad():
            q = (positions.detach() / self.length_factor).requires_grad_(True)
            cell = None if self.cell is None else self.cell.expand(frames, 3, 3)
            strain = None
            qe = q
            if self.virial_mode == "autograd" or (self.virial_mode == "auto" and self.periodic):
                strain = torch.zeros((frames, 3, 3), dtype=q.dtype, device=q.device, requires_grad=True)
                transform = torch.eye(3, dtype=q.dtype, device=q.device) + strain
                qe = q @ transform.transpose(-1, -2)
                if cell is not None:
                    cell = cell @ transform.transpose(-1, -2)
            out = self.model(qe, cell, self.pbc.expand(frames, 3)) if self.periodic else self.model(qe)
            force = virial = stress = None
            if isinstance(out, dict):
                energy = out.get("energy", out.get("energies"))
                force, virial, stress = out.get("forces"), out.get("virial"), out.get("stress")
                if virial is not None and stress is not None:
                    raise ValueError("return either virial or stress, not both")
            elif isinstance(out, (tuple, list)):
                if len(out) != 2:
                    raise ValueError("model tuple must contain (energy, forces)")
                energy, force = out
            else:
                energy = out
            if not isinstance(energy, Tensor) or energy.shape not in ((frames,), (frames, 1)):
                raise ValueError("energy must be a tensor [frames] or [frames,1]")
            energy = energy.reshape(frames)
            _tensor(energy, "energy", (frames,), self.device)
            if stress is not None:
                _tensor(stress, "stress", (frames, 3, 3), self.device)
                if self._volume is None:
                    raise ValueError("stress requires a cell")
                virial = -stress * self._volume
            derive_strain = strain is not None and (self.virial_mode == "autograd" or virial is None)
            targets = ([q] if force is None else []) + ([strain] if derive_strain else [])
            if targets:
                if not energy.requires_grad:
                    raise ValueError("detached model energy cannot supply forces/virial")
                gradients = torch.autograd.grad(energy.sum(), targets, allow_unused=True)
                if all(g is None for g in gradients):
                    raise ValueError("model energy is disconnected from coordinates/strain")
                gradients = iter(gradients)
                if force is None:
                    gradient = next(gradients)
                    force = torch.zeros_like(q) if gradient is None else -gradient
                if derive_strain:
                    gradient = next(gradients)
                    virial = torch.zeros_like(strain) if gradient is None else -gradient
            _tensor(force, "forces", positions.shape, self.device)
            if self.virial_mode == "provided" and virial is None:
                raise ValueError("virial='provided' requires virial or stress")
            if self.virial_mode == "none":
                virial = None
            if virial is not None:
                _tensor(virial, "virial", (frames, 3, 3), self.device)
                virial = virial.detach() * self.energy_factor
            return TensorPESResult(energy.detach() * self.energy_factor,
                                   force.detach() * (self.energy_factor / self.length_factor), virial)


class TorchRPCTerm:
    """One additive energy component on Pc contracted beads, across all rings.

    Optional ``atoms`` select the coordinates on which THIS energy component
    depends. Interactions with atoms outside that set must be separate energy
    components; masking forces of an undecomposed model would be incorrect.
    Energy is distributed uniformly over P beads when Pc<P; only its sum is
    defined. Force lifting is the exact derivative (P/Pc) T.T F.
    """
    def __init__(self, potential: TorchPES, nbeads, contracted_beads=None, *, atoms=None):
        if not isinstance(potential, TorchPES):
            raise TypeError("potential must be a TorchPES")
        self.potential, self.device = potential, potential.device
        self.nbeads = _integer(nbeads, "nbeads")
        self.contracted_beads = self.nbeads if contracted_beads is None else _integer(contracted_beads, "contracted_beads")
        if self.contracted_beads > self.nbeads:
            raise ValueError("contracted_beads cannot exceed nbeads")
        self.atoms = None
        self._indices = None
        if atoms is not None:
            self.atoms = tuple(_integer(i, "atom index", 0) for i in atoms)
            if not self.atoms or len(set(self.atoms)) != len(self.atoms):
                raise ValueError("atoms must be nonempty and distinct")
            self._indices = torch.tensor(self.atoms, dtype=torch.long, device=self.device)
        p, c = self.nbeads, self.contracted_beads
        if c == p:
            matrix = torch.eye(p, dtype=torch.float64, device=self.device)
        else:
            source = torch.arange(p, dtype=torch.float64, device=self.device)[None, :] / p
            target = torch.arange(c, dtype=torch.float64, device=self.device)[:, None] / c
            matrix = torch.ones((c, p), dtype=torch.float64, device=self.device)
            for k in range(1, (c + 1) // 2):
                matrix += 2 * torch.cos(2 * math.pi * k * (target - source))
            if c % 2 == 0:
                matrix += math.sqrt(2) * torch.cos(math.pi * torch.arange(c, dtype=torch.float64, device=self.device))[:, None] * torch.cos(math.pi * c * source)
            matrix /= p
        self._matrix = matrix

    @property
    def matrix(self):
        """A copy of the coordinate projection, for inspection."""
        return self._matrix.clone()

    def evaluate_tensor(self, q):
        if not isinstance(q, Tensor) or q.ndim != 4 or min(q.shape) < 1:
            raise ValueError("q must be nonempty [trajectories,beads,atoms,3]")
        b, _, n, _ = q.shape
        p, c = self.nbeads, self.contracted_beads
        _tensor(q, "q", (b, p, n, 3), self.device)
        if self.atoms is not None and max(self.atoms) >= n:
            raise ValueError("regional atom index out of range")
        selected = q if self._indices is None else q.index_select(2, self._indices)
        contracted = selected if c == p else torch.einsum("cj,bjna->bcna", self._matrix, selected)
        result = self.potential.evaluate_tensor(contracted.reshape(b * c, selected.shape[2], 3))
        energy = result.energy.reshape(b, c)
        forces = result.forces.reshape(b, c, selected.shape[2], 3)
        if c != p:
            energy = energy.mean(dim=1, keepdim=True).expand(b, p)
            forces = (p / c) * torch.einsum("cj,bcna->bjna", self._matrix, forces)
        if self._indices is not None:
            full = torch.zeros_like(q)
            full.index_copy_(2, self._indices, forces)
            forces = full
        return TensorPESResult(energy, forces)


class TorchSplitPotential:
    """Explicit fast and slow additive terms; no implicit full-minus-reference.

    Supply correction models for slow terms when fast is a reference model.
    Each term's model is called once for all trajectories and contracted beads.
    Components remain caller-owned; this object does not close them.
    """
    def __init__(self, fast: Sequence[TorchRPCTerm], slow: Sequence[TorchRPCTerm] = ()):
        self.fast, self.slow = tuple(fast), tuple(slow)
        terms = self.fast + self.slow
        if not self.fast or not all(isinstance(t, TorchRPCTerm) for t in terms):
            raise ValueError("fast must contain at least one TorchRPCTerm; slow must contain terms")
        self.device, self.nbeads = terms[0].device, terms[0].nbeads
        if any(t.device != self.device or t.nbeads != self.nbeads for t in terms):
            raise ValueError("all terms must have the same device and full bead count")

    @staticmethod
    def _evaluate(terms, q):
        # Own the accumulators: components may share/reuse model output storage.
        energy = torch.zeros(q.shape[:2], dtype=q.dtype, device=q.device)
        force = torch.zeros_like(q)
        for term in terms:
            result = term.evaluate_tensor(q)
            energy.add_(result.energy)
            force.add_(result.forces)
        return TensorPESResult(energy, force)

    def evaluate_fast(self, q):
        return self._evaluate(self.fast, q)

    def evaluate_slow(self, q):
        return self._evaluate(self.slow, q)

    def evaluate_tensor(self, q):
        return self._evaluate(self.fast + self.slow, q)


@dataclass
class TorchState:
    """Owned detached q,p tensors [trajectories,beads,atoms,3]."""
    q: Tensor
    p: Tensor
    step: int = 0
    time: float = 0.0

    def __post_init__(self):
        if not isinstance(self.q, Tensor) or self.q.ndim != 4 or min(self.q.shape) < 1:
            raise ValueError("q,p must be nonempty tensors [trajectories,beads,atoms,3]")
        b, p, n, _ = self.q.shape
        _tensor(self.q, "q", (b, p, n, 3), self.q.device)
        _tensor(self.p, "p", self.q.shape, self.q.device)
        if not (torch.isfinite(self.q).all() & torch.isfinite(self.p).all()).item():
            raise ValueError("q,p must be finite")
        self.step = _integer(self.step, "step", 0)
        self.time = _positive(self.time, "time", allow_zero=True)
        self.q = self.q.detach().clone(memory_format=torch.contiguous_format)
        self.p = self.p.detach().clone(memory_format=torch.contiguous_format)

    @classmethod
    def from_arrays(cls, q, p, *, device="cpu", step=0, time=0.0):
        """Explicit upload boundary; accepts [B,P,N,3] or one [P,N,3] ring."""
        raw_q = torch.as_tensor(q, device=device)
        raw_p = torch.as_tensor(p, device=device)
        if any(value.is_complex() or value.dtype == torch.bool for value in (raw_q, raw_p)):
            raise ValueError("q,p must contain real numerical values, not complex or boolean data")
        # Convert from the original objects: converting Python floats through
        # a default-float32 temporary would irreversibly lose precision.
        q = torch.as_tensor(q, dtype=torch.float64, device=device)
        p = torch.as_tensor(p, dtype=torch.float64, device=device)
        if q.ndim == 3 and p.ndim == 3:
            q, p = q.unsqueeze(0), p.unsqueeze(0)
        return cls(q, p, step, time)

    def snapshot(self):
        """Explicit host-copy boundary, not a legacy workflow checkpoint."""
        return {"q": self.q.detach().cpu().numpy().copy(),
                "p": self.p.detach().cpu().numpy().copy(),
                "step": self.step, "time": self.time}


class TorchDynamics:
    """Batched exact-normal-mode/Cayley propagation, PILE-L/TRPMD, and MTS.

    dt and State.step count OUTER steps. The deterministic map is
    Bslow(dt/2) [Bfast(h/2) A(h) Bfast(h/2)]**M Bslow(dt/2), h=dt/M.
    Optional OU thermostat half-steps bracket that map (OBABO/OBCBO).
    ``integrator="cayley-bcocb"`` instead applies B(dt/2) C(dt)**(1/2)
    O(dt) C(dt)**(1/2) B(dt/2), with a principal square-root Cayley map,
    NOT a Cayley map at half dt (Korol et al., JCP 152, 104102, Eq. 17).
    B includes all fast/slow/RPC terms; this option requires inner_steps=1.
    Its harmonic configurational marginal is exact within the physical-force
    stability limit, but stored momenta have finite-step bias. Use position-
    based quantum estimators; energy() remains the mechanical Hamiltonian.
    Production
    dynamics is unthermostatted unless explicitly requested. PILE-L thermalizes
    the centroid; TRPMD leaves it untouched. Real-mass variance is m*P/beta.

    ``normal_mode_layout="rpmdrate"`` uses RPMDrate's FFTW halfcomplex
    packing and 1/sqrt(P) normalization, independently of the integrator.
    The BCOCB C/O/C sequence then stays in normal modes between half maps.
    The default ``"rfft"`` preserves the original complex-rFFT arithmetic.
    Neither layout changes real masses, Hamiltonian units or PES batching.

    A propagate block is atomic, including RNG rollback on failure. Finite
    status is accumulated on device and read once at the block boundary.
    There is no persistent force cache across blocks or external state edits.
    """
    def __init__(self, potential, masses, beta, nbeads, dt, *, inner_steps=1,
                 integrator="normal-mode-bab", thermostat=None, gamma_centroid=1.0,
                 seed=0, normal_mode_layout="rfft"):
        if normal_mode_layout not in ("rfft", "rpmdrate"):
            raise ValueError("normal_mode_layout must be rfft or rpmdrate")
        self.normal_mode_layout = normal_mode_layout
        self.nbeads = _integer(nbeads, "nbeads")
        self.inner_steps = _integer(inner_steps, "inner_steps")
        self.beta, self.dt = _positive(beta, "beta"), _positive(dt, "dt")
        if isinstance(potential, TorchPES):
            potential = TorchSplitPotential([TorchRPCTerm(potential, self.nbeads)])
        if not isinstance(potential, TorchSplitPotential) or potential.nbeads != self.nbeads:
            raise ValueError("potential must be a TorchPES or matching TorchSplitPotential")
        if self.inner_steps > 1 and not potential.slow:
            raise ValueError("inner_steps>1 requires an explicit slow potential component")
        if integrator not in ("normal-mode-bab", "cayley-bcb", "cayley-bcocb"):
            raise ValueError("integrator must be normal-mode-bab, cayley-bcb, or cayley-bcocb")
        if integrator == "cayley-bcocb" and self.inner_steps != 1:
            raise ValueError("cayley-bcocb requires inner_steps=1; BCOCB/MTS is not implemented")
        if thermostat not in (None, "pile-l", "trpmd"):
            raise ValueError("thermostat must be None, pile-l, or trpmd")
        self.potential, self.device = potential, potential.device
        self.integrator, self.thermostat = integrator, thermostat
        self.gamma_centroid = _positive(gamma_centroid, "gamma_centroid", allow_zero=True)
        raw_masses = torch.as_tensor(masses, device=self.device)
        if raw_masses.is_complex() or raw_masses.dtype == torch.bool:
            raise ValueError("masses must be real numerical values")
        self._masses = torch.as_tensor(masses, dtype=torch.float64, device=self.device).detach().clone()
        if (self._masses.ndim != 1 or not self._masses.numel()
                or not (torch.isfinite(self._masses).all() & (self._masses > 0).all()).item()):
            raise ValueError("masses must be a nonempty, finite, positive vector")
        self._mass = self._masses.reshape(1, 1, -1, 1)
        self._sigma = torch.sqrt(self._mass * (self.nbeads / self.beta))
        k = torch.arange(self.nbeads // 2 + 1, dtype=torch.float64, device=self.device)
        omega = 2 * self.nbeads / self.beta * torch.sin(math.pi * k / self.nbeads)
        self._normal_modes = None
        if normal_mode_layout == "rpmdrate":
            from .torch_normal_modes import RPMDrateNormalModes
            self._normal_modes = RPMDrateNormalModes(self.nbeads, device=self.device)
            omega = omega[self._normal_modes.frequency_indices]
        h = self.dt / self.inner_steps
        if integrator == "normal-mode-bab":
            theta = h * omega
            a, b, c = torch.cos(theta), h * torch.sinc(theta / math.pi), omega * torch.sin(theta)
        else:
            z = h * omega / 2
            den = 1 + z.square()
            a, b, c = (1 - z.square()) / den, h / den, h * omega.square() / den
        self._abc = tuple(v.reshape(1, -1, 1, 1) for v in (a, b, c))
        self._half_abc = None
        if integrator == "cayley-bcocb":
            # Principal square root of cay(dt*A). Squaring gives _abc, including
            # the exact centroid drift. cay(dt*A/2) would be a DIFFERENT method.
            root_den = torch.sqrt(1 + (h * omega / 2).square())
            self._half_abc = tuple(v.reshape(1, -1, 1, 1) for v in
                                   (1 / root_den, (h / 2) / root_den,
                                    (h / 2) * omega.square() / root_den))
        gamma = 2 * omega
        gamma[0] = self.gamma_centroid if thermostat == "pile-l" else 0
        ou_dt = self.dt if integrator == "cayley-bcocb" else self.dt / 2
        self._decay = torch.exp(-gamma * ou_dt).reshape(1, -1, 1, 1)
        self._noise = torch.sqrt(-torch.expm1(-gamma * (2 * ou_dt))).reshape(1, -1, 1, 1)
        if not all(torch.isfinite(v).all().item() for v in
                   (*self._abc, *(self._half_abc or ()), self._sigma, self._decay, self._noise)):
            raise ValueError("free-ring or thermostat coefficients overflow")
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(_integer(seed, "seed", 0))
        self.force_evaluations = {"fast": 0, "slow": 0}
        self._parameters = self._signature()

    def _signature(self):
        return (self.nbeads, self.inner_steps, self.beta, self.dt, self.integrator,
                self.thermostat, self.gamma_centroid, id(self.potential), self.normal_mode_layout)

    def _check(self, state):
        if not isinstance(state, TorchState):
            raise TypeError("state must be a TorchState")
        if self._signature() != self._parameters:
            raise ValueError("dynamics parameters are immutable; construct a new engine")
        if not isinstance(state.q, Tensor) or state.q.ndim != 4 or state.q.shape[0] < 1:
            raise ValueError("state must contain a nonempty trajectory batch")
        shape = (state.q.shape[0], self.nbeads, self._masses.numel(), 3)
        _tensor(state.q, "q", shape, self.device)
        _tensor(state.p, "p", shape, self.device)
        if not state.q.is_contiguous() or not state.p.is_contiguous():
            raise ValueError("state tensors must be contiguous and nonoverlapping")
        if state.q.requires_grad or state.p.requires_grad:
            raise ValueError("state tensors must be detached")
        if state.q.untyped_storage().data_ptr() == state.p.untyped_storage().data_ptr():
            raise ValueError("q,p must not share storage")
        _integer(state.step, "step", 0)
        _positive(state.time, "time", allow_zero=True)

    def sample_momenta(self, trajectories=1):
        shape = (_integer(trajectories, "trajectories"), self.nbeads, self._masses.numel(), 3)
        return torch.randn(shape, dtype=torch.float64, device=self.device, generator=self.generator) * self._sigma

    def _free(self, q, p, *, half=False):
        if self._normal_modes is not None:
            modes = self._normal_modes
            qk, pk = modes.forward(q, bead_axis=1), modes.forward(p, bead_axis=1)
            a, b, c = self._half_abc if half else self._abc
            return (modes.inverse(a * qk + b * pk / self._mass, bead_axis=1),
                    modes.inverse(a * pk - c * self._mass * qk, bead_axis=1))
        qk = torch.fft.rfft(q, dim=1, norm="ortho")
        pk = torch.fft.rfft(p, dim=1, norm="ortho")
        a, b, c = self._half_abc if half else self._abc
        return (torch.fft.irfft(a * qk + b * pk / self._mass, n=self.nbeads, dim=1, norm="ortho"),
                torch.fft.irfft(a * pk - c * self._mass * qk, n=self.nbeads, dim=1, norm="ortho"))

    def _thermostat(self, p):
        if self.thermostat is None:
            return p
        # Transform REAL bead noise. Independently sampling complex Fourier
        # noise gives incorrect centroid/Nyquist variances and correlations.
        noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=self.generator)
        if self._normal_modes is not None:
            modes = self._normal_modes
            pk, nk = modes.forward(p, bead_axis=1), modes.forward(noise, bead_axis=1)
            return modes.inverse(self._decay * pk + self._noise * self._sigma * nk, bead_axis=1)
        pk = torch.fft.rfft(p, dim=1, norm="ortho")
        nk = torch.fft.rfft(noise, dim=1, norm="ortho")
        return torch.fft.irfft(self._decay * pk + self._noise * self._sigma * nk,
                               n=self.nbeads, dim=1, norm="ortho")

    def _middle(self, q, p):
        """C^(1/2) O C^(1/2); packed path stays in normal modes throughout."""
        if self._normal_modes is None:
            q, p = self._free(q, p, half=True)
            p = self._thermostat(p)
            return self._free(q, p, half=True)
        modes = self._normal_modes
        qk, pk = modes.forward(q, bead_axis=1), modes.forward(p, bead_axis=1)
        a, b, c = self._half_abc
        qk, pk = a * qk + b * pk / self._mass, a * pk - c * self._mass * qk
        # Same real bead noise and RNG sequence as the default layout. Its
        # transform has variance 1 at zero/Nyquist and 1/2 in each paired slot.
        noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=self.generator)
        nk = modes.forward(noise, bead_axis=1)
        pk = self._decay * pk + self._noise * self._sigma * nk
        qk, pk = a * qk + b * pk / self._mass, a * pk - c * self._mass * qk
        return modes.inverse(qk, bead_axis=1), modes.inverse(pk, bead_axis=1)

    def step(self, state):
        return self.propagate(state, 1)

    @torch.no_grad()
    def propagate(self, state, steps):
        steps = _integer(steps, "steps", 0)
        self._check(state)
        if steps == 0:
            return state
        final_time = state.time + steps * self.dt
        if not math.isfinite(final_time):
            raise ValueError("state clock overflow")
        q, p = state.q.clone(), state.p.clone()
        rng_state = self.generator.get_state()
        valid = torch.isfinite(q).all() & torch.isfinite(p).all()

        def force(kind):
            nonlocal valid
            self.force_evaluations[kind] += 1
            evaluator = self.potential.evaluate_fast if kind == "fast" else self.potential.evaluate_slow
            result = evaluator(q)
            valid = valid & torch.isfinite(result.energy).all() & torch.isfinite(result.forces).all()
            return result.forces.clone()

        try:
            fast = force("fast")
            slow = force("slow") if self.potential.slow else None
            h = self.dt / self.inner_steps
            middle_ou = self.integrator == "cayley-bcocb"
            for _ in range(steps):
                if not middle_ou:
                    p = self._thermostat(p)
                if slow is not None:
                    p.add_(slow, alpha=self.dt / 2)
                for _ in range(self.inner_steps):
                    p.add_(fast, alpha=h / 2)
                    if middle_ou and self.thermostat is not None:
                        q, p = self._middle(q, p)
                    else:
                        # With no thermostat the two square roots collapse
                        # to the existing full C map, without extra FFTs.
                        q, p = self._free(q, p)
                    fast = force("fast")
                    p.add_(fast, alpha=h / 2)
                if slow is not None:
                    slow = force("slow")
                    p.add_(slow, alpha=self.dt / 2)
                if not middle_ou:
                    p = self._thermostat(p)
                valid = valid & torch.isfinite(q).all() & torch.isfinite(p).all()
            if not valid.item():
                raise FloatingPointError("nonfinite model result or integration state; block rolled back")
        except BaseException:
            self.generator.set_state(rng_state)
            raise
        state.q.copy_(q)
        state.p.copy_(p)
        state.step += steps
        state.time = final_time
        return state

    @torch.no_grad()
    def energy(self, state):
        """Return one sum-beads Hamiltonian per trajectory, still on device."""
        self._check(state)
        result = self.potential.evaluate_tensor(state.q)
        kinetic = (state.p.square() / (2 * self._mass)).sum(dim=(1, 2, 3))
        dq = state.q - torch.roll(state.q, -1, dims=1)
        spring = 0.5 * (self.nbeads / self.beta) ** 2 * (self._mass * dq.square()).sum(dim=(1, 2, 3))
        return result.energy.sum(dim=1) + kinetic + spring


__all__ = ["TensorPESResult", "TorchPES", "TorchRPCTerm", "TorchSplitPotential", "TorchState", "TorchDynamics"]
