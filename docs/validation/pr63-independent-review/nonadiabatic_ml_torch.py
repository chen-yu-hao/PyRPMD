"""Fresh CPU TorchScript committee training, monitoring and RPC integration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from pyrpmd.dynamics import Dynamics, State
from pyrpmd.potentials.builtin import HarmonicPotential
from pyrpmd.potentials.lifecycle import ActiveLearningMonitor, CommitteePotential, DeltaLearningPotential
from pyrpmd.potentials.optional import TorchPotential


OUTPUT = Path(__file__).resolve().parent / "nonadiabatic_ml" / "torch"
SEEDS = [64129, 64139, 64151, 64153]


class LearnedCorrection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(1, 12), torch.nn.Tanh(),
                                       torch.nn.Linear(12, 12), torch.nn.Tanh(),
                                       torch.nn.Linear(12, 1))

    def forward(self, q):
        return self.net(q[:, 0, 0:1]).squeeze(-1)


def main():
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    x = torch.linspace(-1., 1., 48)
    q = torch.zeros((len(x), 1, 3))
    q[:, 0, 0] = x
    labels = .05*x**4
    files, losses = [], []
    for i, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        model = LearnedCorrection()
        indices = torch.randint(len(x), (len(x),))
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        for _ in range(600):
            optimizer.zero_grad()
            loss = torch.mean((model(q[indices])-labels[indices])**2)
            loss.backward()
            optimizer.step()
        losses.append(float(torch.mean((model(q)-labels)**2).detach()))
        model.eval()
        path = OUTPUT/f"member_{i}.pt"
        torch.jit.trace(model, q).save(str(path))
        files.append(path)
    np.savez_compressed(OUTPUT/"training.npz", q=q.numpy(), correction_energy=labels.numpy(), seeds=SEEDS)
    members = [TorchPotential(model=str(path), device="cpu") for path in files]
    committee = CommitteePotential(members)
    heldout = np.zeros((101, 1, 3))
    heldout[:,0,0] = np.linspace(-2.5, 2.5, len(heldout))
    prediction = committee.evaluate_uncertainty(heldout)
    truth_e = .05*heldout[:,0,0]**4
    truth_f = -.2*heldout[:,0,0]**3
    individual = np.stack([member.evaluate(heldout).forces for member in members])
    disagreement_error = float(np.max(abs(prediction.force_std[:,0]-individual[:, :, 0, 0].std(axis=0))))
    assert disagreement_error < 2e-15
    np.savez_compressed(OUTPUT/"heldout.npz", q=heldout, truth_energy=truth_e,
        truth_force=truth_f, predicted_energy=prediction.mean.energy,
        predicted_force=prediction.mean.forces[:,0,0], force_std=prediction.max_force_std)

    # Exercise actual propagation with and without monitoring on the same
    # deterministic serialized models.  The force path is unchanged.
    initial = np.zeros((4,1,3))
    initial[:,0,0] = np.array([1.6,1.8,2.,2.2])
    p = np.zeros_like(initial)
    with_monitor = Dynamics(committee, [2.], 2., 4, .01, native=False, seed=7123)
    without_monitor = Dynamics(committee, [2.], 2., 4, .01, native=False, seed=7123)
    monitored, control = State(initial,p), State(initial,p)
    collection = OUTPUT/"collection"
    if collection.exists():
        shutil.rmtree(collection)
    identity = hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest()
    monitor = ActiveLearningMonitor(committee, collection, force_threshold=.005,
                                   interval_steps=5, max_frames=10, model_identity=identity)
    trace = []
    for _ in range(50):
        with_monitor.propagate(monitored, 1, record_centroids=False)
        without_monitor.propagate(control, 1, record_centroids=False)
        monitor.observe(monitored.q, step=monitored.step, time=monitored.time)
        trace.append(monitored.q.copy())
    assert np.array_equal(monitored.q, control.q) and np.array_equal(monitored.p, control.p)
    assert monitor.collected == 10
    assert len(list(collection.glob("step_*.npz"))) == 10
    labels_available = []
    for path in collection.glob("step_*.npz"):
        with np.load(path, allow_pickle=False) as record:
            labels_available.append(bool(record["labels_available"]))
    assert not any(labels_available)
    np.savez_compressed(OUTPUT/"monitored_trajectory.npz", q=np.asarray(trace), p_final=monitored.p)

    # The learned delta uses the same trained mean and its autograd derivative.
    delta = DeltaLearningPotential(HarmonicPotential(k=.4), committee)
    split = delta.split(4, 1)
    result = split.evaluate(initial)
    h = 2e-6
    error = 0.
    for index in np.ndindex(initial.shape):
        plus, minus = initial.copy(), initial.copy()
        plus[index] += h
        minus[index] -= h
        force = -(split.evaluate(plus).energy.sum()-split.evaluate(minus).energy.sum())/(2*h)
        error = max(error, abs(force-result.forces[index]))
    assert error < 1e-8
    inside = abs(heldout[:,0,0]) <= 1
    summary = dict(torch=torch.__version__, seeds=SEEDS, training_steps_per_model=600,
        energy_training_mse=losses, models_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        force_rmse_inside=float(np.sqrt(np.mean((prediction.mean.forces[inside,0,0]-truth_f[inside])**2))),
        force_rmse_outside=float(np.sqrt(np.mean((prediction.mean.forces[~inside,0,0]-truth_f[~inside])**2))),
        committee_force_std_error=disagreement_error, monitored_steps=50,
        trajectories_bitwise_equal=True, candidates_collected=monitor.collected,
        all_candidates_unlabelled=True, learned_rpc_force_fd_error=float(error),
        elapsed_seconds=time.perf_counter()-start,
        limitations=["Synthetic noiseless quartic correction, not molecular or ab initio training.",
                     "Small finite-training networks are imperfect, especially outside their training range.",
                     "No labeling/retraining loop is executed; candidate records remain unlabeled.",
                     "No assertion that committee disagreement bounds the true error."])
    root = Path(__file__).resolve().parents[3]
    sources = [Path(__file__), root/"src/pyrpmd/dynamics.py",
               root/"src/pyrpmd/potentials/optional.py", root/"src/pyrpmd/potentials/lifecycle.py",
               root/"src/pyrpmd/potentials/contraction.py"]
    summary["source_sha256"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sources}
    (OUTPUT/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))
    for member in members:
        member.close()


if __name__ == "__main__":
    main()
