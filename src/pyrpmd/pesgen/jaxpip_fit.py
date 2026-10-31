"""Offline energy/force fitting of real JaxPIP linear and neural-network models.

Training uses only the explicit training split. Neither validation nor test
points determine normalization, optimization, stopping or checkpoint selection.
JAX is never imported on the production C ABI dynamics path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import json
import importlib.metadata
from pathlib import Path
import shutil
import tempfile
import tomllib

import numpy as np
from scipy.optimize import minimize

from ..constants import BOHR_ANGSTROM, EV_TO_AU
from .jaxpip_data import (SPLITS, checked_json, load_dataset, member, output_lock,
                          write_checked)
from .reference import atomic_json, canonical_hash, file_hash


@dataclass(frozen=True)
class FitSettings:
    kind: str = "linear"
    alpha: float = 1.0
    position_unit: str = "bohr"
    energy_unit: str = "hartree"
    energy_scale: float = 0.01  # hartree, independent of model units
    force_scale: float = 0.01  # hartree/bohr
    force_weight: float = 1.0
    ridge: float = 1e-10  # linear only; not applied to constant coefficient
    hidden_layers: tuple[int, ...] = (16, 16)
    activation: str = "tanh"
    seed: int = 0
    max_steps: int = 500
    gradient_tolerance: float = 1e-8
    max_design_elements: int = 20_000_000

    def __post_init__(self):
        if self.kind not in ("linear", "nn") or self.activation not in ("tanh", "isru"):
            raise ValueError("kind must be linear/nn and activation tanh/isru")
        if self.position_unit not in ("bohr", "angstrom") or self.energy_unit not in ("hartree", "eV"):
            raise ValueError("model units must be bohr/angstrom and hartree/eV")
        for name in ("alpha", "energy_scale", "force_scale", "gradient_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("force_weight", "ridge"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("seed", "max_steps", "max_design_elements"):
            if type(getattr(self, name)) is not int or getattr(self, name) < (0 if name == "seed" else 1):
                raise ValueError(f"invalid integer {name}")
        if self.seed > 2**32 - 1:
            raise ValueError("seed must fit uint32")
        if not isinstance(self.hidden_layers, (tuple, list)) or not self.hidden_layers or any(type(x) is not int or x < 1 for x in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive integers")
        object.__setattr__(self, "hidden_layers", tuple(self.hidden_layers))

    @property
    def conversions(self):
        # q_model = L*q_au; E_model = U*E_au; F_model = U/L*F_au.
        return (1.0 if self.position_unit == "bohr" else BOHR_ANGSTROM,
                1.0 if self.energy_unit == "hartree" else 1.0 / EV_TO_AU)


def validate_basis(path, natoms):
    # Upstream stores exponents as uint8: reject values which would wrap.
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        basis = json.load(stream)
    ndist = natoms*(natoms-1)//2
    if not isinstance(basis, list) or len(basis) < 2 or basis[0] != [[0]*ndist]:
        raise ValueError("basis needs a leading constant and at least one nonconstant feature")
    for polynomial in basis:
        if not isinstance(polynomial, list) or not polynomial:
            raise ValueError("empty/invalid basis polynomial")
        for monomial in polynomial:
            if (not isinstance(monomial, list) or len(monomial) != ndist
                    or any(type(v) is not int or not 0 <= v <= 255 for v in monomial)):
                raise ValueError("basis monomials need N(N-1)/2 integer exponents in [0,255]")


def _jax():
    try:
        import jax
        import equinox as eqx
        from jaxpip.descriptor import PolynomialDescriptor
        from jaxpip.model import PolynomialLinearModel, PolynomialNeuralNetwork
    except ImportError as exc:
        raise ImportError("Offline fitting needs the optional JaxPIP/JAX/Equinox environment; see the JaxPIP generation guide") from exc
    jax.config.update("jax_enable_x64", True)
    return jax, eqx, PolynomialDescriptor, PolynomialLinearModel, PolynomialNeuralNetwork


def _arrays(points, settings):
    length, energy = settings.conversions
    return (np.stack([p.positions for p in points]) * length,
            np.array([p.energy for p in points]) * energy,
            -np.stack([p.gradient for p in points]) * energy / length)


def _fit_model(descriptor, q, e, f, settings):
    jax, eqx, _, Linear, Network = _jax()
    import jax.numpy as jnp
    from jax.flatten_util import ravel_pytree
    q, e, f = map(jnp.asarray, (q, e, f))
    m, n = q.shape[:2]
    nf = descriptor.feature_dim
    needed = m * nf * (1 + (3*n if settings.force_weight else 0)) + nf*nf
    if needed > settings.max_design_elements:
        raise ValueError("descriptor/design exceeds max_design_elements; reduce the basis or dataset")
    features = np.asarray(jax.jit(jax.vmap(descriptor))(q))
    if not np.isfinite(features).all() or not np.allclose(features[:, 0], 1., atol=1e-12, rtol=0):
        raise ValueError("JaxPIP basis must have a finite leading constant feature equal to one")
    length, energy = settings.conversions
    escale, fscale = settings.energy_scale*energy, settings.force_scale*energy/length
    if settings.kind == "linear":
        offset = float(jnp.mean(e))
        a = features / (escale*np.sqrt(m))
        b = (np.asarray(e)-offset) / (escale*np.sqrt(m))
        if settings.force_weight:
            jac = np.asarray(jax.jit(jax.vmap(jax.jacrev(descriptor)))(q))
            # Match JaxPIP's removal of roundoff-level net translation forces.
            fb = -jac + jac.mean(axis=2, keepdims=True)
            fb = fb.transpose(0, 2, 3, 1).reshape(-1, nf)
            factor = np.sqrt(settings.force_weight)/(fscale*np.sqrt(f.size))
            a = np.concatenate((a, fb*factor))
            b = np.concatenate((b, np.asarray(f).ravel()*factor))
        if not np.isfinite(a).all():
            raise ValueError("nonfinite descriptor derivatives")
        if settings.ridge:
            penalty = np.eye(nf) * np.sqrt(settings.ridge)
            penalty[0, 0] = 0.
            a = np.concatenate((a, penalty))
            b = np.concatenate((b, np.zeros(nf)))
        coeffs, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
        coeffs[0] += offset
        model = Linear(descriptor, coeffs=jnp.asarray(coeffs))
        diagnostics = {"optimizer": "weighted augmented SVD least squares", "rank": int(rank),
                       "features": nf, "regularization": settings.ridge,
                       "singular_values": singular.tolist()}
    else:
        model = Network(descriptor, list(settings.hidden_layers), jax.random.PRNGKey(settings.seed),
                        activation=settings.activation)
        # Only train points set both feature and output scales.
        model = model.update_scaler(jnp.asarray(features), e)
        if float(jnp.ptp(e)) < 1e-14:
            raise ValueError("NN training needs a nonzero training-energy range")
        params, static = eqx.partition(model.layers, eqx.is_inexact_array)
        vector, unravel = ravel_pytree(params)

        def objective(vector):
            layers = eqx.combine(unravel(vector), static)
            candidate = eqx.tree_at(lambda x: x.layers, model, layers)
            if settings.force_weight:
                pe, pf = jax.vmap(candidate.get_energy_and_forces)(q)
                return jnp.mean(((pe-e)/escale)**2) + settings.force_weight*jnp.mean(((pf-f)/fscale)**2)
            return jnp.mean(((jax.vmap(candidate.get_energy)(q)-e)/escale)**2)

        derivative = jax.jit(jax.value_and_grad(objective))

        def scipy_objective(v):
            value, grad = derivative(jnp.asarray(v))
            value, grad = float(value), np.asarray(grad, dtype=float)
            if not np.isfinite(value) or not np.isfinite(grad).all():
                raise FloatingPointError("nonfinite NN loss/gradient")
            return value, grad

        initial = scipy_objective(np.asarray(vector))[0]
        result = minimize(scipy_objective, np.asarray(vector), jac=True, method="L-BFGS-B",
                          options={"maxiter": settings.max_steps, "gtol": settings.gradient_tolerance,
                                   "ftol": 1e-14, "maxls": 40})
        if not np.isfinite(result.x).all():
            raise FloatingPointError("nonfinite fitted NN weights")
        model = eqx.tree_at(lambda x: x.layers, model, eqx.combine(unravel(jnp.asarray(result.x)), static))
        diagnostics = {"optimizer": "L-BFGS-B on layers only; frozen train-only scaler",
                       "success": bool(result.success), "message": str(result.message),
                       "iterations": int(result.nit), "initial_loss": initial,
                       "final_loss": float(result.fun), "gradient_max": float(np.max(np.abs(result.jac)))}
    return model, diagnostics


def evaluate_metrics(model, points, settings):
    if not points:
        return None
    jax, eqx, *_ = _jax()
    q, e, f = _arrays(points, settings)
    @eqx.filter_jit
    def batch(q):
        return jax.vmap(model.get_energy_and_forces)(q)
    pred_e, pred_f = batch(q)
    pred_e, pred_f = np.asarray(pred_e), np.asarray(pred_f)
    if pred_e.shape != e.shape or pred_f.shape != f.shape or not (np.isfinite(pred_e).all() and np.isfinite(pred_f).all()):
        raise ValueError("fitted model produced invalid energies/forces")
    length, energy = settings.conversions
    de, df = (pred_e-e)/energy, (pred_f-f)*length/energy
    return {"count": len(points), "energy_rmse_hartree": float(np.sqrt(np.mean(de**2))),
            "force_rmse_hartree_per_bohr": float(np.sqrt(np.mean(df**2))),
            "energy_max_hartree": float(np.max(np.abs(de))),
            "force_max_hartree_per_bohr": float(np.max(np.abs(df)))}


def _load_fit(directory, file_digest):
    manifest = checked_json(directory / "model.json", "pyrpmd.jaxpip-fit/1")
    for name, digest in manifest["artifacts"].items():
        if file_digest(member(directory, name)) != digest:
            raise ValueError(f"fitted artifact checksum mismatch: {name}")
    required = {manifest["basis"], "model.eqx", "probe.npy", "metrics.json"}
    if not required.issubset(manifest["artifacts"]):
        raise ValueError("missing fitted model artifacts")
    return manifest


def load_fit(directory):
    return _load_fit(Path(directory).resolve(), file_hash)


def fit_jaxpip(input_file, output, *, resume=False):
    """Fit from a strict TOML declaration; preserve .eqx compatibility upstream."""
    input_file, output = Path(input_file).resolve(), Path(output).resolve()
    if type(resume) is not bool:
        raise ValueError("resume must be Boolean")
    spec = tomllib.loads(input_file.read_text())
    if set(spec) - {"schema", "dataset", "basis", "basis_symbols", "fit", "acceptance"} or spec.get("schema") != "pyrpmd.jaxpip-fit/1":
        raise ValueError("expected strict pyrpmd.jaxpip-fit/1 input")
    settings = FitSettings(**spec.get("fit", {}))
    dataset = (input_file.parent / spec["dataset"]).resolve()
    basis = (input_file.parent / spec["basis"]).resolve()
    data, points = load_dataset(dataset)
    if spec.get("basis_symbols") != data["contract"]["symbols"]:
        raise ValueError("basis_symbols must explicitly match the ordered dataset elements")
    if not (basis.name.endswith(".json") or basis.name.endswith(".json.gz")):
        raise ValueError("basis must be a JaxPIP .json or .json.gz file")
    bounds = spec.get("acceptance", {})
    if bounds and set(bounds) != {"energy_rmse_max", "force_rmse_max"}:
        raise ValueError("acceptance requires energy_rmse_max and force_rmse_max in atomic units")
    if any(isinstance(x, bool) or not np.isfinite(x) or x <= 0 for x in bounds.values()):
        raise ValueError("acceptance bounds must be finite and positive")
    jax, _, Descriptor, *_ = _jax()
    import jax.numpy as jnp
    versions = {name: importlib.metadata.version(name) for name in ("jaxpip", "jax", "equinox", "numpy", "scipy")}
    fingerprint = canonical_hash({"spec": spec, "dataset": file_hash(dataset/"dataset.json"),
                                  "basis": file_hash(basis), "versions": versions,
                                  "code": file_hash(Path(__file__)),
                                  "data_code": file_hash(Path(__file__).with_name("jaxpip_data.py"))})
    with output_lock(output):
        if (output/"model.json").exists():
            existing = load_fit(output)
            if not resume or existing["fingerprint"] != fingerprint:
                raise ValueError("existing fit requires resume=True with identical data/settings/environment")
            return existing
        if any(p.name != ".lock" for p in output.iterdir()):
            raise ValueError("refusing nonempty incomplete fit; choose a fresh output directory")
        validate_basis(basis, len(data["contract"]["symbols"]))
        descriptor = Descriptor.from_file(str(basis), alpha=settings.alpha, dtype=jnp.float64)
        if descriptor.basis_info.num_atoms != len(data["contract"]["symbols"]):
            raise ValueError("basis atom count does not match the ordered dataset")
        q, e, f = _arrays(points["train"], settings)
        # A basis has no chemical labels: the caller declares their ordering.
        # Test identical-element generating swaps on deterministic training probes.
        descriptor_fn = jax.jit(jax.vmap(descriptor))
        probes = q[:min(4, len(q))]
        ref_features = np.asarray(descriptor_fn(probes))
        groups = {}
        for i, symbol in enumerate(data["contract"]["symbols"]):
            groups.setdefault(symbol, []).append(i)
        for indices in groups.values():
            for a, b in zip(indices, indices[1:]):
                permuted = probes.copy()
                permuted[:, [a, b]] = permuted[:, [b, a]]
                if not np.allclose(np.asarray(descriptor_fn(permuted)), ref_features, atol=1e-11, rtol=1e-10):
                    raise ValueError("basis is not invariant to declared identical-element permutations")
        model, optimizer = _fit_model(descriptor, q, e, f, settings)
        metrics = {s: evaluate_metrics(model, points[s], settings) for s in SPLITS}
        test = metrics["test"]
        accepted = bool(bounds and test and test["energy_rmse_hartree"] <= bounds["energy_rmse_max"]
                        and test["force_rmse_hartree_per_bohr"] <= bounds["force_rmse_max"])
        report = {"metrics": metrics, "optimizer": optimizer, "acceptance": bounds,
                  "within_requested_test_bounds": accepted, "physical_validation": False,
                  "split_usage": "train only for fit/scaler/stopping; validation/test are reports only",
                  "scope": "supplied molecular domain; not global extrapolation or converged rate validation"}
        # Stage complete artifacts, then commit the manifest last. A partial
        # publication is rejected instead of guessing which checkpoint is valid.
        with tempfile.TemporaryDirectory(prefix="jaxpip-fit-") as td:
            stage = Path(td)
            basisname = "basis.json.gz" if basis.name.endswith(".gz") else "basis.json"
            shutil.copyfile(basis, stage/basisname)
            model.save(str(stage/"model.eqx"))
            np.save(stage/"probe.npy", q[:min(len(q), 8)], allow_pickle=False)
            atomic_json(stage/"metrics.json", report)
            artifacts = {p.name: file_hash(p) for p in stage.iterdir()}
            for p in stage.iterdir():
                shutil.copyfile(p, output/p.name)
        manifest = {"schema": "pyrpmd.jaxpip-fit/1", "fingerprint": fingerprint,
                    "basis": basisname, "checkpoint": "model.eqx", "contract": data["contract"],
                    "settings": asdict(settings), "versions": versions, "artifacts": artifacts,
                    "dataset_sha256": file_hash(dataset/"dataset.json"),
                    "within_requested_test_bounds": accepted, "physical_validation": False}
        write_checked(output/"model.json", manifest)
        return manifest
