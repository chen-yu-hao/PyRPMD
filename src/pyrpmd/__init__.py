"""PyRPMD: ring-polymer dynamics with one Python and .inp task model."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "JobSpec":
        from .input import JobSpec

        return JobSpec
    if name == "System":
        from .system import System

        return System
    if name in ("State", "Dynamics"):
        from . import dynamics

        return getattr(dynamics, name)
    if name == "run":
        from .runner import run

        return run
    raise AttributeError(name)


__all__ = ["JobSpec", "System", "State", "Dynamics", "run", "__version__"]
