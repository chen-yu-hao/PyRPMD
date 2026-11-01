"""BCOCB sampling -> RPMDrate-style exact normal-mode RPMD production.

Teaching oscillator, not a chemistry benchmark. Run with:
  PYTHONPATH=src python examples/ensembles/torch_normal_modes.py --device cpu
CUDA is explicitly selectable; no CPU fallback is made.
"""
import argparse
import json
import torch
from pyrpmd.torch_backend import TorchDynamics, TorchPES, TorchState
from pyrpmd.torch_normal_modes import RPMDrateNormalModes


class Harmonic(torch.nn.Module):
    metadata = {"batch_independent": True}
    def forward(self, q):
        return q.square().sum((1, 2)) / 2, -q


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(1)
    P, beta, dt = 16, 8., .1
    pes = TorchPES(Harmonic(), device=args.device)
    q = torch.zeros((32, P, 1, 3), dtype=torch.float64, device=pes.device)
    state = TorchState(q, q)
    sampling = TorchDynamics(pes, [1.], beta, P, dt, integrator="cayley-bcocb",
                             normal_mode_layout="rpmdrate", thermostat="pile-l", seed=17)
    sampling.propagate(state, 500)
    modes = RPMDrateNormalModes(P, device=pes.device)
    packed = modes.forward(state.q, bead_axis=1)
    restored = modes.inverse(packed, bead_axis=1)
    # Positions are in bohr, momenta in atomic units with real masses. A
    # constrained rate launch would additionally require a constrained Maxwell
    # momentum draw; this unconstrained example does not implement that step.
    production = TorchDynamics(pes, [1.], beta, P, dt, integrator="normal-mode-bab",
                               normal_mode_layout="rpmdrate", thermostat=None, seed=23)
    launch = TorchState(restored, production.sample_momenta(len(state.q)))
    initial_energy = production.energy(launch)
    production.propagate(launch, 100)
    print(json.dumps({
        "device": str(launch.q.device), "trajectories": len(launch.q), "beads": P,
        "roundtrip_max_error": (state.q-restored).abs().max().item(),
        "production_steps": launch.step, "production_thermostat": production.thermostat,
        "max_relative_energy_change": ((production.energy(launch)/initial_energy)-1).abs().max().item(),
    }, indent=2))


if __name__ == "__main__":
    main()
