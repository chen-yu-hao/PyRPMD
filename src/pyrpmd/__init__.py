"""PyRPMD: ring-polymer dynamics with one Python and .inp task model."""

__version__ = "0.2.0"


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
    if name in ("run", "prepare", "resume_run", "analyze_run"):
        from . import runner

        return getattr(runner, name)
    raise AttributeError(name)


__all__ = [
    "JobSpec", "System", "State", "Dynamics", "run", "prepare",
    "resume_run", "analyze_run", "__version__",
]
