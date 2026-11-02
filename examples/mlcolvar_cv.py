"""Train mlcolvar Deep-LDA/Deep-TICA and export a PyRPMD TorchScript CV.

Run from a checkout with ``python examples/mlcolvar_cv.py --method deep-tica``.
Optional dependencies: ``pip install 'mlcolvar>=1.2,<2'``.  Training and export
are explicitly float64 on CPU.  Small synthetic datasets demonstrate the API;
they are not evidence of a converged molecular reaction coordinate.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

import numpy as np


def _training_data(torch, method, n_samples, seed):
    """Return float64 data in bohr, with explicit time pairing for Deep-TICA."""
    rng = np.random.default_rng(seed)
    if method == "deep-lda":
        labels = np.arange(n_samples) % 2
        centers = np.array([[-0.8, 0.4], [0.8, -0.3]])
        data = centers[labels] + rng.normal(scale=0.25, size=(n_samples, 2))
        return {"data": torch.as_tensor(data, dtype=torch.float64),
                "labels": torch.as_tensor(labels, dtype=torch.int64)}
    # Two reversible stationary OU processes with distinct relaxation times.
    # These are uniformly sampled unbiased data, so time-pair weights are 1.
    decay = np.array([0.97, 0.45])
    path = np.empty((n_samples + 1, 2), dtype=np.float64)
    path[0] = rng.normal(size=2)
    for index in range(n_samples):
        path[index + 1] = decay * path[index] + np.sqrt(1 - decay**2) * rng.normal(size=2)
    return {"data": torch.as_tensor(path[:-1], dtype=torch.float64),
            "data_lag": torch.as_tensor(path[1:], dtype=torch.float64),
            "weights": torch.ones(n_samples, dtype=torch.float64),
            "weights_lag": torch.ones(n_samples, dtype=torch.float64)}


def train_and_export(output, *, method="deep-tica", n_samples=512, epochs=20, seed=2026):
    """Train a genuine mlcolvar model and export positions[N,3] -> CV[1].

    The toy descriptors are x and y of atom zero in bohr.  The exported graph
    includes descriptor extraction, input normalization, the trained network
    and the learned LDA/TICA projection.  PyRPMD must not apply normalization
    again.  No model is moved to a different device or precision implicitly.
    Returns the trained descriptor model and the TorchScript path for checks.
    """
    if method not in ("deep-lda", "deep-tica"):
        raise ValueError("method must be deep-lda or deep-tica")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 64:
        raise ValueError("n_samples must be an integer >=64")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    try:
        import torch
        import lightning
        from mlcolvar.cvs import DeepLDA, DeepTICA
        from mlcolvar.data import DictDataset, DictModule
        from packaging.specifiers import SpecifierSet
    except ImportError as exc:
        raise ImportError("this example requires torch, lightning and mlcolvar>=1.2,<2") from exc
    if not SpecifierSet(">=1.2,<2").contains(version("mlcolvar"), prereleases=False):
        raise ValueError("this example targets mlcolvar>=1.2,<2; select a supported training environment")

    # mlcolvar's eigensolvers allocate scratch tensors in the default dtype.
    # Set it explicitly during construction/training and restore the caller's
    # choice on exit.  Trainer's precision rule also prevents mixed precision.
    previous_dtype = torch.get_default_dtype()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    with torch.random.fork_rng(devices=[]):
        try:
            torch.set_default_dtype(torch.float64)
            # This example trains only on CPU.  manual_seed() would also
            # change accelerator generators outside this CPU-only fork.
            torch.random.default_generator.manual_seed(seed)
            tensors = _training_data(torch, method, n_samples, seed)
            dataset = DictDataset(tensors)
            # Full batch statistics avoid classes disappearing from LDA batches.
            # A shuffled split is suitable only for this smoke demonstration;
            # molecular time series need separated trajectory/block holdouts.
            datamodule = DictModule(dataset, lengths=[0.8, 0.2], batch_size=0)
            options = {"nn": {"activation": "tanh"}, "optimizer": {"lr": 1e-3}}
            if method == "deep-lda":
                model = DeepLDA([2, 8, 2], n_states=2, options=options)
            else:
                model = DeepTICA([2, 8, 2], n_cvs=1, options=options)
                model.set_regularization(c0_reg=1e-5)
            model = model.to(device="cpu", dtype=torch.float64)
            trainer = lightning.Trainer(
                accelerator="cpu", devices=1, precision="64-true", max_epochs=epochs,
                deterministic=True, logger=False, enable_checkpointing=False,
                enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
            )
            trainer.fit(model, datamodule)
            model.eval()
            # Refit the linear projection on the complete training partition
            # after the last optimizer update.  Validation must not determine
            # which projection is frozen into the deployed CV.
            with torch.no_grad():
                training = next(iter(datamodule.train_dataloader()))
                features = model.forward_nn(training["data"])
                if method == "deep-lda":
                    model.lda.compute(features, training["labels"], save_params=True)
                    projection = model.lda
                else:
                    lagged = model.forward_nn(training["data_lag"])
                    model.tica.compute(data=[features, lagged],
                                       weights=[training["weights"], training["weights_lag"]],
                                       save_params=True)
                    projection = model.tica

            class PositionCV(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.inference = torch.nn.Sequential(model.norm_in, model.nn, projection)

                def forward(self, positions):
                    # Shape is [N,3], not the [batch,descriptor] training shape.
                    return self.inference(positions[0, :2].unsqueeze(0))[0]

            deployment = PositionCV().eval()
            example = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64)
            alternate = torch.tensor([[-0.4, 0.7, 0.0]], dtype=torch.float64)
            scripted = torch.jit.trace(deployment, example, check_inputs=[(alternate,)], strict=True)
            output = Path(output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.jit.save(scripted, str(output))
            # Check the serialized artifact, including coordinate derivatives.
            loaded = torch.jit.load(str(output), map_location="cpu").eval()
            for coordinates in (example, alternate):
                probe = coordinates.detach().clone().requires_grad_(True)
                result = loaded(probe)
                expected = model(probe[0, :2].unsqueeze(0))[0]
                if result.dtype != torch.float64 or tuple(result.shape) != (1,):
                    raise ValueError("export violated float64 [1] CV contract")
                torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
                gradient = torch.autograd.grad(result.sum(), probe)[0]
                if not bool(torch.isfinite(gradient).all()) or not bool(gradient.abs().max() > 0):
                    raise ValueError("exported learned CV has invalid or zero coordinate derivatives")
            return model, output
        finally:
            torch.set_default_dtype(previous_dtype)
            torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("deep-lda", "deep-tica"), default="deep-tica")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("mlcolvar-cv.pt"))
    args = parser.parse_args()
    _, output = train_and_export(args.output, method=args.method, n_samples=args.samples,
                                 epochs=args.epochs, seed=args.seed)
    from pyrpmd.advanced_cv import TorchScriptCV

    cv = TorchScriptCV(output, dimension=1, units=("dimensionless",),
                       periods=(None,), estimator="centroid", device="cpu")
    q = np.array([[[0.2, -0.1, 0.0]], [[0.4, 0.1, 0.0]]], dtype=np.float64)
    evaluated = cv.value_and_jacobian(q)
    print(f"Exported {args.method} model: {output}")
    print(f"Centroid CV: {evaluated.value}; Jacobian shape: {evaluated.jacobian.shape}")
    print("This validates deployment, not physical or training convergence.")


if __name__ == "__main__":
    main()
