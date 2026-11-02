"""Optional native batch providers with NumPy and device tensor interfaces.

The new names deliberately leave the existing ASE-backed ``mace``, ``nequip``
and ``fairchem`` providers and their scientific identities unchanged. Framework
dependencies are lazy: importing this module does not import Torch or a model.
"""

from importlib import import_module


_PROVIDERS = {
    "mace-batched": ("_batched_mace", "MACEBatchedPotential"),
    "nequip-batched": ("_batched_nequip", "NequIPBatchedPotential"),
    "fairchem-batched": ("_batched_fairchem", "FairchemBatchedPotential"),
    "deepmd-batched": ("_batched_deepmd", "DeepMDBatchedPotential"),
    "metatomic": ("_batched_metatomic", "MetatomicBatchedPotential"),
}

BATCHED_ML_PROVIDERS = frozenset(_PROVIDERS)

__all__ = [value[1] for value in _PROVIDERS.values()]


def __getattr__(name):
    for module, cls in _PROVIDERS.values():
        if name == cls:
            value = getattr(import_module(f".{module}", __package__), cls)
            globals()[name] = value
            return value
    raise AttributeError(name)


def make_batched_ml_potential(kind, **kwargs):
    module, cls = _PROVIDERS[kind]
    return getattr(import_module(f".{module}", __package__), cls)(**kwargs)
