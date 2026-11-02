"""Local MACE batch/tensor demonstration; random weights, no chemical claims."""

from __future__ import annotations

import json


def main():
    import torch
    from _random_mace import make_random_mace
    from pyrpmd.potentials import MACEBatchedPotential

    torch.set_num_threads(1)
    model = make_random_mace(dtype="float64")
    q = torch.tensor([[[0., 0., 0.], [1.8, 0., 0.], [-0.6, 1.7, 0.]]],
                     dtype=torch.float64).repeat(8, 1, 1)
    q[:, 1, 0] += torch.arange(8, dtype=torch.float64) * 0.01
    with MACEBatchedPotential(["O", "D", "T"], model, virial=True) as pes:
        out = pes.evaluate_tensor(q)
        reference = pes.evaluate(q.numpy())
        torch.testing.assert_close(out.energy, torch.from_numpy(reference.energy))
        torch.testing.assert_close(out.forces, torch.from_numpy(reference.forces))
        print(json.dumps({
            "provider": pes.metadata["provider"], "device": str(pes.device),
            "dtype": str(pes.dtype), "frames": len(q), "atoms": q.shape[1],
            "energy_hartree": out.energy.tolist(),
            "force_norm_hartree_per_bohr": out.forces.norm().item(),
            "numpy_tensor_agree": True, "trained_model": False,
            "physical_converged": False,
        }, indent=2))


if __name__ == "__main__":
    main()
